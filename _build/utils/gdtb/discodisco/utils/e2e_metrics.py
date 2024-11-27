import io, os, sys
import json
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# read predictions
prediction_dir = sys.argv[1]
corpus = prediction_dir.split('_')[-1].strip('.json')
out_dir = f'{os.sep}'.join(prediction_dir.split('/')[:-1])

y_pred = []
y_gold = []
count = defaultdict(int)
with io.open(prediction_dir, encoding='utf-8') as f:
    lines = f.read().split('\n')
    for line in lines:
        if not line:
            continue
        data = json.loads(line)
        gold_rel = data['gold_relation']
        pred_rel = data['pred_relation']
        count[gold_rel] += 1
        y_pred.append(pred_rel)
        y_gold.append(gold_rel)


# print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_gold, y_pred)))
#
# print('Micro Precision: {:.2f}'.format(precision_score(y_gold, y_pred, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(y_gold, y_pred, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(y_gold, y_pred, average='micro')))

# print('\nClassification Report\n')
# print(classification_report(y_gold, y_pred))

with io.open(out_dir+os.sep+corpus+'.res', 'w', encoding='utf-8') as f:
    f.write(f'***{corpus}\n')
    f.write('\nAccuracy: {:.4f}\n\n'.format(accuracy_score(y_gold, y_pred)))
    f.write('Micro Precision: {:.4f}\n'.format(precision_score(y_gold, y_pred, average='micro')))
    f.write('Micro Recall: {:.4f}\n'.format(recall_score(y_gold, y_pred, average='micro')))
    f.write('Micro F1-score: {:.4f}\n'.format(f1_score(y_gold, y_pred, average='micro')))
    f.write('\n')
    f.write(classification_report(y_gold, y_pred))
