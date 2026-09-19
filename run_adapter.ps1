param(
    [ValidateSet("smoke-test", "extract", "discover", "train", "predict", "info")]
    [string]$Action = "info",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdapterArgs
)

$ManifoldRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ManifoldRoot ".venv\Scripts\python.exe"
$env:HF_HOME = Join-Path $ManifoldRoot "artifacts\hf_cache"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Environment is missing. Expected $Python"
}

switch ($Action) {
    "info" {
        & $Python -c "import torch, transformers, sklearn; print('Environment ready'); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Transformers:', transformers.__version__); print('scikit-learn:', sklearn.__version__)"
    }
    "smoke-test" { & $Python (Join-Path $ManifoldRoot "smoke_test.py") @AdapterArgs }
    "extract" { & $Python (Join-Path $ManifoldRoot "extract_embeddings.py") @AdapterArgs }
    "discover" { & $Python (Join-Path $ManifoldRoot "discover_manifold.py") @AdapterArgs }
    "train" { & $Python (Join-Path $ManifoldRoot "train_adapter.py") @AdapterArgs }
    "predict" { & $Python (Join-Path $ManifoldRoot "predict.py") @AdapterArgs }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
