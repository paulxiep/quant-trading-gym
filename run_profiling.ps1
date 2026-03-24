# Profiling script for parallelization benchmarking
# Runs the simulation with different parallelization configurations
# 3 trials per config, 1000 ticks each

$NUM_TICKS = 1000
$NUM_TRIALS = 10
$LOG_FILE = "profiling_results.csv"

# Configurations to test (false = sequential for that phase)
$configs = @{
    "all_parallel" = @{}
    "seq_agent_collection" = @{ "PAR_AGENT_COLLECTION" = "false" }
    "seq_indicators" = @{ "PAR_INDICATORS" = "false" }
    "seq_order_validation" = @{ "PAR_ORDER_VALIDATION" = "false" }
    "seq_auctions" = @{ "PAR_AUCTIONS" = "false" }
    "seq_candle_updates" = @{ "PAR_CANDLE_UPDATES" = "false" }
    "seq_trade_updates" = @{ "PAR_TRADE_UPDATES" = "false" }
    "seq_fill_notifications" = @{ "PAR_FILL_NOTIFICATIONS" = "false" }
    "seq_wake_conditions" = @{ "PAR_WAKE_CONDITIONS" = "false" }
    "seq_risk_tracking" = @{ "PAR_RISK_TRACKING" = "false" }
}

Write-Host "=== Parallelization Profiling Benchmark ==="
Write-Host "Ticks per run: $NUM_TICKS"
Write-Host "Trials per config: $NUM_TRIALS"
Write-Host ""

# Build release binary
Write-Host "Building release binary..."
cargo build --release --all-features
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!"
    exit 1
}

# Create CSV header
"config_name,trial,elapsed_ms,ticks_per_sec" | Out-File -FilePath $LOG_FILE -Encoding UTF8

foreach ($configName in $configs.Keys) {
    Write-Host "Testing: $configName"

    for ($trial = 1; $trial -le $NUM_TRIALS; $trial++) {
        # Set environment variables for this config
        foreach ($key in $configs[$configName].Keys) {
            [Environment]::SetEnvironmentVariable($key, $configs[$configName][$key], "Process")
        }

        # Run simulation and measure time
        $start = Get-Date
        $output = & ".\target\release\quant-trading-gym.exe" --headless --ticks $NUM_TICKS 2>&1
        $elapsed = (Get-Date) - $start

        $elapsedMs = [int]$elapsed.TotalMilliseconds
        $ticksPerSec = if ($elapsedMs -gt 0) { ($NUM_TICKS * 1000.0) / $elapsedMs } else { 0.0 }

        # Log results
        "$configName,$trial,$elapsedMs,$([math]::Round($ticksPerSec, 2))" | Out-File -FilePath $LOG_FILE -Append -Encoding UTF8

        Write-Host ("  Trial {0}: {1:F2}s ({2:F1} ticks/s)" -f $trial, $elapsed.TotalSeconds, $ticksPerSec)

        # Clear environment variables
        foreach ($key in $configs[$configName].Keys) {
            [Environment]::SetEnvironmentVariable($key, $null, "Process")
        }
    }

    Write-Host ""
}

Write-Host "Results written to: $LOG_FILE"
