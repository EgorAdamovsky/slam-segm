# Куда класть веса

`run_video.py` читает файлы из каталога `weights/` в корне проекта. В git они не попадают: `.gitignore` игнорирует `*.pt` и `*.pth`.

Скрипту нужны два файла:

```text
weights/rf-detr-seg-preview.pt    сегментация, RF-DETR Seg Preview, около 131 МБ
weights/banet-2d/kitti.pth        глубина, BANet 2D, обучен на KITTI, около 22 МБ
```

Остальное можно положить рядом и передать флагом `--depth`. Само оно не подхватывается.

```text
weights/banet-2d/sceneflow.pth    BANet 2D, Scene Flow, около 22 МБ
weights/banet-3d/kitti.pth        BANet 3D, около 15 МБ, run_video.py его не грузит
weights/banet-3d/sceneflow.pth
```

Для дороги берётся `banet-2d/kitti.pth`. Scene Flow — общий синтетический датасет. Трёхмерный BANet в этом скрипте не используется.

## Сегментация

Прямая ссылка на публичный COCO-чекпоинт (91 класс):

https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt

```bash
mkdir -p weights
curl -L --retry 5 -o weights/rf-detr-seg-preview.pt \
  https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt
```

Готовый файл около 131 МБ. Обрывок на сотни килобайт — это недокачка, его надо заменить. Свой чекпоинт (например Cityscapes, около 30 классов) кладётся в тот же путь или указывается так: `--seg /путь/к/файлу.pt`.

## Глубина

Официальные веса BANet, папка Google Drive (2D и 3D, KITTI и Scene Flow):

https://drive.google.com/drive/folders/1PNnTUP4ntYgzBpkvDqnhmxMJGUisqKvz

Скачивание папки из браузера даёт zip. Внутри уже правильные пути: `banet-2d/kitti.pth`, `banet-2d/sceneflow.pth` и то же для `banet-3d`. Распаковка из корня проекта:

```bash
mkdir -p weights
unzip banet-2d-*.zip -d weights
unzip banet-3d-*.zip -d weights
```

Должны появиться `weights/banet-2d/kitti.pth` (около 22 МБ) и, если скачан второй архив, `weights/banet-3d/kitti.pth` (около 15 МБ).

Отдельный файл MobileNetV2 качать не нужно. Эти веса уже внутри чекпоинта BANet. Пакет `timm` ставится как библиотека, из неё только собирается архитектура `mobilenetv2_100`.

Другой чекпоинт:

```bash
python run_video.py --left left.png --right right.png --depth weights/banet-2d/sceneflow.pth
```

## Запуск

На этой машине интерпретатор с torch и opencv:

`/home/psu320/Egor/SMOKE-DETR/env/bin/python`

Нужны пакеты: `torch`, `torchvision`, `opencv-python`, `timm`, `transformers`, `pydantic`, `peft`.

```bash
python run_video.py --video video.mp4 --max-frames 3 --out model_results --device cuda
python run_video.py --left left.png --right right.png --out model_results --device cuda
```

Без `--left`/`--right` (или `--layout sbs`/`tb`) глубина не считается: один ролик — это моно.
