from mlflow.tracking import MlflowClient

client = MlflowClient("http://127.0.0.1:5000")
versions = client.search_model_versions("name='yandex_maps_sentiment'")
for v in sorted(versions, key=lambda x: int(x.version)):
    run = client.get_run(v.run_id)
    acc = run.data.metrics.get("accuracy", "n/a")
    name = run.data.params.get("config_name", run.info.run_name)
    print(f"v{v.version:>2} | {v.current_stage:12} | accuracy={acc} | {name}")
