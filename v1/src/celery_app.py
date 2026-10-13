import os
import socket
from celery import Celery  # type: ignore[import-untyped]

is_test = os.environ.get("CELERY_ALWAYS_EAGER", "0") == "1"
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _is_redis_available() -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        res = s.connect_ex(("127.0.0.1", 6379))
        s.close()
        return res == 0
    except Exception:
        return False


has_redis = _is_redis_available()
run_eager = is_test or not has_redis

broker_url = "memory://" if run_eager else redis_url
backend_url = "rpc://" if run_eager else redis_url

celery_app = Celery(
    "v1_render_tasks",
    broker=broker_url,
    backend=backend_url,
    include=["src.tasks"],
)

# We use pickle because the result dictionary contains raw numpy arrays (e.g. img_array, disparity).
celery_app.conf.update(
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle", "json"],
    task_always_eager=run_eager,
    task_store_eager_result=run_eager,
)
