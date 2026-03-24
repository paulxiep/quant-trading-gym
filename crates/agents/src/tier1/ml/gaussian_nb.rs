//! Gaussian Naive Bayes model for classification (V6.2).
//!
//! Loads sklearn GaussianNB from JSON and performs inference using
//! precomputed log-Gaussian parameters for zero transcendental function overhead.
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "model_type": "gaussian_nb",
//!   "model_name": "naive_bayes_v6",
//!   "n_features": 55,
//!   "n_classes": 3,
//!   "classes": [-1, 0, 1],
//!   "class_log_prior": [-1.2, -0.8, -1.1],
//!   "theta": [[...55 means...], [...], [...]],
//!   "var": [[...55 variances...], [...], [...]]
//! }
//! ```
//!
//! # Variance Smoothing
//!
//! Matches sklearn's `var_smoothing=1e-9` to prevent division by zero when a
//! feature has zero variance for a class. Variance is clamped at load time and
//! two precomputed arrays eliminate per-prediction `ln()` and division:
//! - `neg_half_log_var[class][i] = -0.5 * ln(var)`
//! - `inv_2var[class][i] = 1 / (2 * var)`
//!
//! Prediction reduces to two multiply-adds per feature per class.

use std::path::Path;

use serde::Deserialize;

use super::{ClassProbabilities, MlModel, softmax};

/// Minimum variance floor, matching sklearn's `var_smoothing` default.
const VAR_SMOOTHING: f64 = 1e-9;

/// JSON deserialization format for sklearn GaussianNB.
#[derive(Debug, Deserialize)]
struct GaussianNBJson {
    model_type: String,
    model_name: String,
    n_features: usize,
    n_classes: usize,
    #[allow(dead_code)]
    classes: Vec<i32>,
    /// Log of class priors: ln(P(class)).
    class_log_prior: Vec<f64>,
    /// Per-class feature means: [n_classes][n_features].
    theta: Vec<Vec<f64>>,
    /// Per-class feature variances: [n_classes][n_features].
    var: Vec<Vec<f64>>,
}

/// Gaussian Naive Bayes classifier loaded from sklearn JSON export.
///
/// Uses precomputed reciprocals and log-variances for fast inference
/// with zero transcendental function calls at predict time.
#[derive(Debug)]
pub struct GaussianNBPredictor {
    /// Model name (e.g., "GaussianNB_naive_bayes_v6").
    name: String,
    /// Log class priors: [3].
    class_log_prior: [f64; 3],
    /// Per-class feature means: [3][n_features].
    theta: Vec<Vec<f64>>,
    /// Precomputed -0.5 * ln(var): [3][n_features].
    neg_half_log_var: Vec<Vec<f64>>,
    /// Precomputed 1/(2*var): [3][n_features].
    inv_2var: Vec<Vec<f64>>,
    /// Number of features expected.
    n_features: usize,
}

impl GaussianNBPredictor {
    /// Load a Gaussian NB predictor from a JSON file.
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        Self::from_json_str(&content)
    }

    /// Load a Gaussian NB predictor from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let parsed: GaussianNBJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

        // Validate model type
        if parsed.model_type != "gaussian_nb" {
            return Err(format!(
                "Expected model_type 'gaussian_nb', got '{}'",
                parsed.model_type
            ));
        }

        // Validate class count
        if parsed.n_classes != 3 {
            return Err(format!("Expected 3 classes, got {}", parsed.n_classes));
        }

        // Validate dimensions
        if parsed.class_log_prior.len() != 3 {
            return Err(format!(
                "Expected 3 class_log_prior values, got {}",
                parsed.class_log_prior.len()
            ));
        }
        if parsed.theta.len() != 3 {
            return Err(format!("Expected 3 theta rows, got {}", parsed.theta.len()));
        }
        if parsed.var.len() != 3 {
            return Err(format!("Expected 3 var rows, got {}", parsed.var.len()));
        }
        for class in 0..3 {
            if parsed.theta[class].len() != parsed.n_features {
                return Err(format!(
                    "theta[{}] has {} elements, expected {}",
                    class,
                    parsed.theta[class].len(),
                    parsed.n_features
                ));
            }
            if parsed.var[class].len() != parsed.n_features {
                return Err(format!(
                    "var[{}] has {} elements, expected {}",
                    class,
                    parsed.var[class].len(),
                    parsed.n_features
                ));
            }
        }

        // Precompute with variance smoothing (matches sklearn var_smoothing=1e-9)
        let mut neg_half_log_var = Vec::with_capacity(3);
        let mut inv_2var = Vec::with_capacity(3);
        for class in 0..3 {
            let mut nhlv = Vec::with_capacity(parsed.n_features);
            let mut i2v = Vec::with_capacity(parsed.n_features);
            for i in 0..parsed.n_features {
                let var = parsed.var[class][i].max(VAR_SMOOTHING);
                nhlv.push(-0.5 * var.ln());
                i2v.push(1.0 / (2.0 * var));
            }
            neg_half_log_var.push(nhlv);
            inv_2var.push(i2v);
        }

        let name = format!("GaussianNB_{}", parsed.model_name);

        Ok(Self {
            name,
            class_log_prior: [
                parsed.class_log_prior[0],
                parsed.class_log_prior[1],
                parsed.class_log_prior[2],
            ],
            theta: parsed.theta,
            neg_half_log_var,
            inv_2var,
            n_features: parsed.n_features,
        })
    }
}

impl MlModel for GaussianNBPredictor {
    fn predict(&self, features: &[f64]) -> ClassProbabilities {
        let n = features.len().min(self.n_features);
        let mut log_probs = [0.0f64; 3];

        for (class, log_prob) in log_probs.iter_mut().enumerate() {
            *log_prob = self.class_log_prior[class];
            for (i, &feat) in features.iter().enumerate().take(n) {
                // Two multiply-adds per feature, zero transcendental functions
                *log_prob += self.neg_half_log_var[class][i]
                    - (feat - self.theta[class][i]).powi(2) * self.inv_2var[class][i];
            }
        }

        softmax(&log_probs)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn n_features(&self) -> usize {
        self.n_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nb_json() -> &'static str {
        r#"{
            "model_type": "gaussian_nb",
            "model_name": "test",
            "n_features": 3,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "class_log_prior": [-1.0986, -1.0986, -1.0986],
            "theta": [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0]
            ],
            "var": [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]
            ]
        }"#
    }

    #[test]
    fn test_load_from_json() {
        let model = GaussianNBPredictor::from_json_str(sample_nb_json()).unwrap();
        assert_eq!(model.name(), "GaussianNB_test");
        assert_eq!(model.n_features(), 3);
    }

    #[test]
    fn test_predictions_sum_to_one() {
        let model = GaussianNBPredictor::from_json_str(sample_nb_json()).unwrap();
        let features = [10.0, 0.0, 0.0]; // Close to class 0 mean
        let probs = model.predict(&features);

        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1.0, got {}",
            sum
        );
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_closest_mean_wins() {
        let model = GaussianNBPredictor::from_json_str(sample_nb_json()).unwrap();

        // Features near class 0 (sell) mean [10, 0, 0]
        let probs = model.predict(&[10.0, 0.0, 0.0]);
        assert!(probs[0] > probs[1], "sell should > hold");
        assert!(probs[0] > probs[2], "sell should > buy");

        // Features near class 2 (buy) mean [0, 0, 10]
        let probs = model.predict(&[0.0, 0.0, 10.0]);
        assert!(probs[2] > probs[0], "buy should > sell");
        assert!(probs[2] > probs[1], "buy should > hold");
    }

    #[test]
    fn test_zero_variance_no_nan() {
        // Variance = 0 for all features → clamped to VAR_SMOOTHING
        let json = r#"{
            "model_type": "gaussian_nb",
            "model_name": "zero_var",
            "n_features": 2,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "class_log_prior": [-1.0986, -1.0986, -1.0986],
            "theta": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "var": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        }"#;
        let model = GaussianNBPredictor::from_json_str(json).unwrap();
        let probs = model.predict(&[1.0, 2.0]);

        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1.0 even with zero variance, got {}",
            sum
        );
        assert!(
            probs.iter().all(|p| p.is_finite()),
            "No NaN/Inf with zero variance: {:?}",
            probs
        );
        // Feature values match class 0 means exactly → class 0 should win
        assert!(probs[0] > probs[1], "class 0 should dominate");
    }

    #[test]
    fn test_invalid_model_type() {
        let json = r#"{"model_type": "decision_tree", "model_name": "x", "n_features": 2, "n_classes": 3, "classes": [-1,0,1], "class_log_prior": [0.0,0.0,0.0], "theta": [[0.0,0.0],[0.0,0.0],[0.0,0.0]], "var": [[1.0,1.0],[1.0,1.0],[1.0,1.0]]}"#;
        let result = GaussianNBPredictor::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("gaussian_nb"));
    }

    #[test]
    fn test_wrong_class_count() {
        let json = r#"{"model_type": "gaussian_nb", "model_name": "x", "n_features": 2, "n_classes": 2, "classes": [-1,1], "class_log_prior": [0.0,0.0], "theta": [[0.0,0.0],[0.0,0.0]], "var": [[1.0,1.0],[1.0,1.0]]}"#;
        let result = GaussianNBPredictor::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("3 classes"));
    }

    #[test]
    fn test_negative_variance_clamped() {
        // Negative variance values should be clamped to VAR_SMOOTHING
        let json = r#"{
            "model_type": "gaussian_nb",
            "model_name": "neg_var",
            "n_features": 2,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "class_log_prior": [-1.0986, -1.0986, -1.0986],
            "theta": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "var": [[-1.0, -0.5], [1.0, 1.0], [1.0, 1.0]]
        }"#;
        let model = GaussianNBPredictor::from_json_str(json).unwrap();
        let probs = model.predict(&[1.0, 2.0]);

        assert!(
            probs.iter().all(|p| p.is_finite()),
            "No NaN/Inf with negative variance: {:?}",
            probs
        );
    }
}
