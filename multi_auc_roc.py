# SKELETON OF MULTICLASS AUC AND ROC
# This is not perfect you can take this as start point for your project visualization
# if you had any better method please do comment

# PYTHON IMPORTS ...
from numpy import trapz
import matplotlib.pyplot as plt
import numpy as np

# OTHER CODE OUT HERE

# Get Train/Test Preds
labels_train, train_preds = train_predict(network, train_loader)
labels_test, test_preds = test_predict(network, test_loader)

# Threshold
thresholds = np.array(list(range(0, 100, 1)))/100

# ================ ROC Function for Test ======================


def roc_test():
    roc_points = []
    gds_list = [0, 1, 2]  # Three class to show in ROC

    for gds in gds_list:
        for threshold in thresholds:
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            count = 0
            for prediction in test_preds:
                prediction = prediction[gds]
                actual = labels_test[count]
                if prediction >= threshold:
                    prediction_class = 1
                else:
                    prediction_class = 0

                if prediction_class == 1 and actual == 1:
                    tp = tp + 1
                elif prediction_class == 0 and actual == 1:
                    fn = fn + 1
                elif prediction_class == 1 and actual == 0:
                    fp = fp + 1
                elif prediction_class == 0 and actual == 0:
                    tn = tn + 1

                count += 1
        #     print(tp,fn,fp,tn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        #     print(threshold, tpr, fpr)
            roc_points.append([precision, recall])

    return roc_points


# =============== TEST ==================
test_roc_point = roc_test()
gds_0_roc = test_roc_point[0:100]
gds_1_roc = test_roc_point[100:200]
gds_2_roc = test_roc_point[200:300]
ROC_COMBINED = np.concatenate((gds_0_roc, gds_1_roc, gds_2_roc), axis=1)

pivot = pd.DataFrame(ROC_COMBINED, columns=[
                     "gds_x_0", "gds_y_0", "gds_x_1", "gds_y_1", "gds_x_2", "gds_y_2"])
pivot['threshold'] = thresholds


# =================ROC CURVE================================
labels = ['gds 0', 'gds 1', 'gds 2']
colors = ['r', 'g', 'b']

plt.plot([0, 1])

plt.plot(pivot.gds_y_0, pivot.gds_x_0, 'o-', color=colors[0], label=labels[0])
plt.plot(pivot.gds_y_1, pivot.gds_x_1, 'o-', color=colors[1], label=labels[1])
plt.plot(pivot.gds_y_2, pivot.gds_x_2, 'o-', color=colors[2], label=labels[2])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# ===================================================================
# AUC  --- accuracy matric
# AUC requires integral calculus
auc_gds_0 = round(abs(np.trapz(pivot.gds_x_0, pivot.gds_y_0)), 4)
auc_gds_1 = round(abs(np.trapz(pivot.gds_x_1, pivot.gds_y_1)), 4)
auc_gds_2 = round(abs(np.trapz(pivot.gds_x_2, pivot.gds_y_2)), 4)

print(auc_gds_0, auc_gds_1, auc_gds_2)
