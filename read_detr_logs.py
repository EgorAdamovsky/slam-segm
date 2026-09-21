import json
from collections import defaultdict
import matplotlib.pyplot as plt
import math
from pathlib import Path

#test_results = []
map_per_class = defaultdict(list)
map95_per_class = defaultdict(list)
losses = defaultdict(list)
bbox_map50 = []
time_strings = []

def sum_times(time_strings):
    total_seconds = 0
    
    for time_str in time_strings:
        h, m, s = map(int, time_str.split(':'))
        total_seconds += h * 3600 + m * 60 + s
    
    hours_f = total_seconds / 3600
    hours = math.floor(hours_f)
    minutes = (hours_f - hours) * 60
    return hours, round(minutes)

with open("trained_models/att5/log.txt") as f:
    for line in f:
        j = json.loads(line)
        for bname in ['loss']:
            for pref in ['train_', 'test_', 'ema_test_']:
                name = pref + bname
                if name in j:
                    losses[name].append(j[name])
        test_results = j['ema_test_results_json']['class_map']
        # test_results = j['test_results_json']['class_map']
        for res in test_results:
            map_per_class[res['class']].append(res['map@50'])
            map95_per_class[res['class']].append(res['map@50:95'])
        test_coco_eval_bbox = j["ema_test_coco_eval_bbox"]
        bbox_map50.append(test_coco_eval_bbox[1])
        time_strings.append(j["epoch_time"])

base = Path("model_results/stats/")
base.mkdir(parents=True, exist_ok=True)

with open(base/"stats.txt", "w") as f:
    print("Эпох:", len(time_strings), file=f)
    h, m = sum_times(time_strings)
    print(f"Время обучения: {h} часов {m} минут", file=f)
    print("lr:\nlr_drop:\nBatch size:\nВидеокарта:\n", file=f)
    #torch.version.__version__

#print(map_per_class)
#print(map95_per_class)
#print(len(losses))
print("classes:", map_per_class.keys())

fig, axs = plt.subplots(3, 1, figsize=(16, 12))
for name, values in losses.items():
    axs[0].plot(range(len(values)), values, label=name)
axs[0].set_title("Loss")
axs[0].legend()
# for cls, values in map_per_class.items():
    # axs[1].plot(range(len(values)), values, label=cls)
for cls in ['car', 'truck', 'person', 'rider', 'bicycle', 'all']:
    values = map_per_class[cls]
    axs[1].plot(range(len(values)), values, label=cls)
axs[1].set_title("mAP50 Масок")
axs[1].legend()

axs[2].set_title("mAP50 Детектора")
axs[2].plot(range(len(bbox_map50)), bbox_map50)

plt.savefig(base/"plot.svg")
#plt.show()
