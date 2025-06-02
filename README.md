
# Setup

```bash
poetry install
```

### Download dataset via DVC

```bash
# задать переменные окружения с read-only ключом
dvc remote modify r2 --local access_key_id 81143b61b82ee32c3b051a76f2110615
dvc remote modify r2 --local secret_access_key 505bbe1d1266f2dc631742045ff52e15ce34b5f169f8c7ada96ba3331974b81c`

# получить данные
dvc pull
```
