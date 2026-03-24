//! Buffered Parquet writer for ML training data.
//!
//! V5.5.2: Simplified to market features only. Agent features removed
//! as tree agents only use market data for inference.
//!
//! Output: `{name}_market.parquet` - Market features (1 row per tick per symbol)

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Buffer size before flushing to Parquet (number of records).
const MARKET_BUFFER_SIZE: usize = 1_000;

// ─────────────────────────────────────────────────────────────────────────────
// Record Types
// ─────────────────────────────────────────────────────────────────────────────

/// Market-level features (1 row per tick per symbol).
#[derive(Debug, Clone)]
pub struct MarketRecord {
    pub tick: u64,
    pub symbol: String,
    /// Market features (42 for V5, 55+ for V6).
    pub features: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Market Parquet Writer (V5.5.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Parquet writer for market features.
///
/// Creates `{base}_market.parquet` with feature columns determined by
/// the provided feature names (42 for V5/MinimalFeatures, 55+ for V6/FullFeatures).
pub struct MarketParquetWriter {
    writer: ParquetTableWriter,
    feature_names: Vec<String>,
}

impl MarketParquetWriter {
    /// Create a new market writer with the given feature names for schema.
    ///
    /// Feature names determine the Parquet column schema. Pass
    /// `MarketFeatures::default_feature_names()` for V5 compatibility, or
    /// `extractor.feature_names()` for a custom extractor.
    pub fn new<P: AsRef<Path>>(
        base_path: P,
        feature_names: &[&str],
    ) -> Result<Self, ParquetWriterError> {
        let base = base_path.as_ref();
        let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("data");
        let parent = base.parent().unwrap_or(Path::new("."));

        let market_path = parent.join(format!("{}_market.parquet", stem));

        // Create parent directories
        std::fs::create_dir_all(parent).map_err(|e| ParquetWriterError::Io(e.to_string()))?;

        let feature_names: Vec<String> = feature_names.iter().map(|s| s.to_string()).collect();
        let schema = Self::build_schema(&feature_names);

        Ok(Self {
            writer: ParquetTableWriter::new(&market_path, schema, MARKET_BUFFER_SIZE)?,
            feature_names,
        })
    }

    fn build_schema(feature_names: &[String]) -> Schema {
        let mut fields = vec![
            Field::new("tick", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
        ];
        for name in feature_names {
            fields.push(Field::new(name, DataType::Float64, true));
        }
        Schema::new(fields)
    }

    /// Write a market record (1 per tick per symbol).
    pub fn write(&mut self, record: MarketRecord) -> Result<(), ParquetWriterError> {
        self.writer.write_market_record(record, &self.feature_names)
    }

    /// Finish writing and close files.
    pub fn finish(self) -> Result<usize, ParquetWriterError> {
        let count = self.writer.finish()?;
        eprintln!("[MarketParquetWriter] Written: {} rows", count);
        Ok(count)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal Table Writer
// ─────────────────────────────────────────────────────────────────────────────

/// Generic Parquet table writer with buffering.
struct ParquetTableWriter {
    schema: Arc<Schema>,
    writer: Option<ArrowWriter<File>>,
    records_written: usize,
    buffer_size: usize,
    current_batch: Option<BatchBuilder>,
}

struct BatchBuilder {
    builders: Vec<ColumnBuilder>,
    num_rows: usize,
}

enum ColumnBuilder {
    UInt64(UInt64Builder),
    String(StringBuilder),
    Float64(Float64Builder),
}

impl ParquetTableWriter {
    fn new<P: AsRef<Path>>(
        path: P,
        schema: Schema,
        buffer_size: usize,
    ) -> Result<Self, ParquetWriterError> {
        let file =
            File::create(path.as_ref()).map_err(|e| ParquetWriterError::Io(e.to_string()))?;
        let schema_arc = Arc::new(schema);

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let writer = ArrowWriter::try_new(file, schema_arc.clone(), Some(props))
            .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;

        Ok(Self {
            schema: schema_arc,
            writer: Some(writer),
            records_written: 0,
            buffer_size,
            current_batch: None,
        })
    }

    fn write_market_record(
        &mut self,
        record: MarketRecord,
        feature_names: &[String],
    ) -> Result<(), ParquetWriterError> {
        let batch = self
            .current_batch
            .get_or_insert_with(|| BatchBuilder::new_market(self.buffer_size, feature_names.len()));

        // tick
        if let ColumnBuilder::UInt64(b) = &mut batch.builders[0] {
            b.append_value(record.tick);
        }
        // symbol
        if let ColumnBuilder::String(b) = &mut batch.builders[1] {
            b.append_value(&record.symbol);
        }
        // features (declarative iteration)
        record.features.iter().enumerate().for_each(|(i, &val)| {
            if let ColumnBuilder::Float64(b) = &mut batch.builders[2 + i] {
                if val.is_nan() {
                    b.append_null();
                } else {
                    b.append_value(val);
                }
            }
        });

        batch.num_rows += 1;
        if batch.num_rows >= self.buffer_size {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ParquetWriterError> {
        let batch = match self.current_batch.take() {
            Some(b) if b.num_rows > 0 => b,
            _ => return Ok(()),
        };

        let columns: Vec<ArrayRef> = batch.builders.into_iter().map(|b| b.finish()).collect();
        let record_batch = RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| ParquetWriterError::Arrow(e.to_string()))?;

        if let Some(ref mut writer) = self.writer {
            writer
                .write(&record_batch)
                .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;
        }

        self.records_written += batch.num_rows;
        Ok(())
    }

    fn finish(mut self) -> Result<usize, ParquetWriterError> {
        self.flush()?;
        if let Some(writer) = self.writer.take() {
            writer
                .close()
                .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;
        }
        Ok(self.records_written)
    }
}

impl BatchBuilder {
    fn new_market(capacity: usize, num_features: usize) -> Self {
        let mut builders = Vec::with_capacity(2 + num_features);
        builders.push(ColumnBuilder::UInt64(UInt64Builder::with_capacity(
            capacity,
        )));
        builders.push(ColumnBuilder::String(StringBuilder::with_capacity(
            capacity,
            capacity * 16,
        )));
        for _ in 0..num_features {
            builders.push(ColumnBuilder::Float64(Float64Builder::with_capacity(
                capacity,
            )));
        }
        Self {
            builders,
            num_rows: 0,
        }
    }
}

impl ColumnBuilder {
    fn finish(self) -> ArrayRef {
        match self {
            Self::UInt64(mut b) => Arc::new(b.finish()),
            Self::String(mut b) => Arc::new(b.finish()),
            Self::Float64(mut b) => Arc::new(b.finish()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during Parquet writing.
#[derive(Debug)]
pub enum ParquetWriterError {
    Io(String),
    Parquet(String),
    Arrow(String),
}

impl std::fmt::Display for ParquetWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {}", msg),
            Self::Parquet(msg) => write!(f, "Parquet error: {}", msg),
            Self::Arrow(msg) => write!(f, "Arrow error: {}", msg),
        }
    }
}

impl std::error::Error for ParquetWriterError {}
