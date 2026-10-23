# Лабораторная 2: очистка данных и удалённая классификация

Работа выполняется вручную на двух Linux-хостах в одной сети. Узел А готовит CSV и отправляет его по SSH/SCP. Узел Б отдельно обучает модель, затем отдельно запускает Flask; узел А запрашивает сохранённые проверочные строки. Команды ниже выполняйте в обычном терминале, не в ячейках ноутбука.

## 1. Подготовить оба хоста

Скопируйте проект на оба хоста, установите Docker Engine и Docker Compose v2. На каждом хосте самостоятельно настройте SSH-сервер и создайте его host keys (на Linux обычно `sudo ssh-keygen -A`). В конфигурации sshd обоих хостов задайте:

```text
PermitRootLogin no
MaxAuthTries 3
```

Проверьте конфигурацию и перезапустите SSH-службу (имя службы зависит от дистрибутива):

```bash
sudo sshd -t
sudo systemctl restart ssh
sudo systemctl status ssh
```

Создайте пользовательскую пару ключей на хосте А, добавьте публичный ключ пользователю хоста Б и проверьте вход:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/lab2_ed25519
ssh-copy-id -i ~/.ssh/lab2_ed25519.pub USER_B@IP_B
ssh -i ~/.ssh/lab2_ed25519 USER_B@IP_B true
```

Замените `USER_B` и `IP_B` своими значениями; подтвердите только проверенный отпечаток хоста Б. Приватный ключ остаётся в `~/.ssh` на хосте А и в контейнер не монтируется.

На хосте Б из корня проекта создайте каталог назначения и запомните его абсолютный путь:

```bash
mkdir -p node_b/data
realpath node_b/data
```

На обоих хостах проверьте сетевые интерфейсы командой `ip a`, а состояние Docker — `sudo systemctl status docker`. Проверьте, что адреса хостов доступны в обе стороны командой `ping`. Она проверяет ICMP, а не TCP-порт Flask. Для Wi-Fi проверьте, что клиентская изоляция (AP/client isolation) отключена. Windows/WSL использует отдельную маршрутизацию портов; одного `ipconfig` недостаточно, чтобы гарантировать доступ между хостами.

## 2. Запустить контейнеры

На хосте Б из корня проекта:

```bash
cd node_b
cp .env.example .env
```

Команду `cp` выполняйте только при первой настройке, чтобы не перезаписать уже настроенный `.env`. В файле оставьте `TARGET_COLUMN=species`, `TEST_SIZE=0.2`, `RANDOM_STATE=42`. Затем соберите образ и запустите контейнер:

```bash
docker compose up --build -d
docker ps --filter name=ml-trainer
```

На хосте А из корня проекта:

```bash
cd node_a
cp .env.example .env
```

Также только при первой настройке. Убедитесь, что указано `TARGET_COLUMN=species`, затем:

```bash
docker compose up --build -d
docker ps --filter name=data-processor
```

Оба контейнера пока простаивают. Сборка устанавливает зависимости и копирует скрипты в `/app`; контейнеры не запускают подготовку данных, обучение или сервер автоматически. Данные узла А хранятся в `node_a/data`, данные для чтения узлом Б монтируются в `/home/DATA` только для чтения, а артефакты модели сохраняются в `node_b/artifacts` и переживают `down`/`up`.

После пересоздания контейнера Flask нужно снова запустить вручную. Повторное обучение сохранённой модели не требуется.

## 3. Подготовить и очистить CSV на хосте А

Подготовка встроенного Iris — отдельная явная команда. Она экспортирует Iris из `sklearn` в `/home/DATA/raw_data.csv` (это не сетевое скачивание) и откажется перезаписывать существующий файл:

```bash
docker exec data-processor python /app/download_dataset.py
```

Только если вы намеренно заменяете существующий `raw_data.csv`, выполните вместо основной команды:

```bash
docker exec data-processor python /app/download_dataset.py --overwrite
```

Вместо этой команды можно положить выданный CSV в `node_a/data/raw_data.csv` на хосте А, либо скопировать его в контейнер командой `docker cp`. Затем отдельно выполните очистку:

```bash
docker exec data-processor python /app/clean_data.py
```

Скрипт читает `/home/DATA/raw_data.csv` и пишет `/home/DATA/clean_data.csv`; он не скачивает данные и не передаёт их по SSH.

## 4. Передать очищенный файл и обучить модель на хосте Б

На хосте А, находясь в корне проекта, вручную выполните SCP:

```bash
scp -i ~/.ssh/lab2_ed25519 node_a/data/clean_data.csv USER_B@IP_B:/ABS/PATH/project/node_b/data/clean_data.csv
```

Подставьте пользователя и IP хоста Б, а также абсолютный путь к каталогу назначения, который показал `realpath node_b/data` на хосте Б. Если текущий каталог терминала уже `node_a`, используйте исходный путь `data/clean_data.csv`. Перед копированием убедитесь, что файл существует на хосте А. Это передача между хостами; Docker-том на хосте Б читает `node_b/data`.

На хосте Б проверьте каталог и файл, затем запустите обучение вручную:

```bash
cd /ABS/PATH/project
realpath node_b/data
ls -l node_b/data/clean_data.csv
docker exec ml-trainer python /app/train_model.py
```

Обучение использует стратифицированное разбиение 80/20, обучает pipeline только на обучающей части, сохраняет `/home/MODEL/model.joblib` (pipeline и метаданные) и `/home/MODEL/test_data.csv` (признаки и истинный класс), затем завершается. Для Iris это 120 обучающих строк и 30 проверочных. `TARGET_COLUMN` должен совпадать в настройках узлов А и Б; для своего размеченного CSV задайте имя целевого столбца на обоих узлах. Поддерживаются числовые и категориальные табличные признаки. При переобучении повторите эту команду вручную после передачи нового CSV.

Методичка требует, чтобы предписанные команды `sudo docker exec` выполнялись без запроса пароля. Наш сценарий не вызывает удалённый Docker через SSH: SCP выполняет студент на хосте А, а `docker exec` — студент локально на нужном хосте. Если sudoers настройка необходима в вашей среде, редактируйте её через `sudo visudo` и ограничьте доступ требуемой командой; не выдавайте unrestricted доступ ко всем командам Docker.

## 5. Запустить Flask и запросить проверочные строки

После успешного обучения на хосте Б в отдельном терминале запустите сервер:

```bash
docker exec -it ml-trainer python /app/ml_service.py
```

Flask слушает `0.0.0.0:5000` в foreground, поэтому терминал занят, пока сервер работает. `Ctrl+C` останавливает Flask, контейнер остаётся запущенным и простаивающим. `/predict` только выполняет inference на сохранённых тестовых строках и возвращает accuracy; он не обучает модель и не делает новое разбиение. До появления артефактов `/predict` отвечает `503`. `/health` показывает статус и готовность модели и тестовых данных.

На хосте А запросите проверочные строки (замените `IP_B`):

```bash
docker exec data-processor curl http://IP_B:5000/health
docker exec data-processor curl http://IP_B:5000/predict
```

Успешный `/health` сообщает `status: "ok"`, `model_ready` и `test_data_ready`. Успешный `/predict` сообщает `status: "success"`, `train_rows: 120`, `test_rows: 30`, `accuracy` и 30 объектов `predictions` с полями `row_index` и `prediction`. `row_index` — индекс строки в очищенном CSV, начиная с 0.

## 6. Ограничить порт 5000 на хосте Б

Сначала проверьте текущие правила. Не вставляйте дубликаты, если правила уже настроены:

```bash
sudo iptables -t nat -nvL
sudo iptables -nvL DOCKER-USER --line-numbers
```

В цепочке `DOCKER-USER` разрешите адрес хоста А первым правилом, а затем запретите остальные входящие TCP-соединения к 5000. Подставьте IP хоста А:

```bash
sudo iptables -I DOCKER-USER 1 -s IP_A -p tcp --dport 5000 -j ACCEPT
sudo iptables -I DOCKER-USER 2 -p tcp --dport 5000 -j DROP
sudo iptables -nvL DOCKER-USER --line-numbers
sudo netfilter-persistent save
```

При необходимости установите пакет `iptables-persistent` средствами своего дистрибутива. Проверьте `/health` и `/predict` с хоста А и недоступность порта с другого устройства.

## 7. Отчёт и ноутбуки

`node_a/data_processor.ipynb` и `node_b/ml_trainer.ipynb` — пошаговые шпаргалки с командами для ручного выполнения; они сами не управляют Docker. Четыре скрипта: `/app/download_dataset.py` экспортирует Iris в `/home/DATA/raw_data.csv`; `/app/clean_data.py` читает raw CSV и создаёт `/home/DATA/clean_data.csv`; `/app/train_model.py` сохраняет `/home/MODEL/model.joblib` и `/home/MODEL/test_data.csv`; `/app/ml_service.py` запускает Flask и использует сохранённые артефакты. На хосте Б эти артефакты находятся в `node_b/artifacts` и сохраняются после перезапуска контейнера. `row_index` в ответе API — индекс строки очищенного CSV, начиная с 0. В отчёте студент вручную описывает скрипты и добавляет требуемые преподавателем скриншоты выполнения команд и результатов. Проверки двух физических хостов, SSH, сети и firewall следует выполнять на самих хостах.

## Необязательная локальная проверка

Для программной проверки на одном компьютере из корня проекта используйте локальный Compose-файл. Он применяет те же обычные Dockerfiles и idle-контейнеры; подготовка, очистка, копирование, обучение и Flask запускаются вручную. Копирование локальное и не проверяет SSH:

```bash
docker compose -f compose.local.yaml up --build -d
docker exec data-processor-local python /app/download_dataset.py
docker exec data-processor-local python /app/clean_data.py
cp node_a/data/clean_data.csv node_b/data/clean_data.csv
docker exec ml-trainer-local python /app/train_model.py
docker exec -it ml-trainer-local python /app/ml_service.py
```

Пока сервер работает, откройте другой терминал и выполните запрос из контейнера А:

```bash
docker exec data-processor-local curl http://ml-trainer-local:5000/predict
```

Локальный порт Flask привязан к `127.0.0.1:5000`. Остановите сервер `Ctrl+C`, затем остановите контейнеры:

```bash
docker compose -f compose.local.yaml down
```
