import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_utils import *
from sklearn.metrics import precision_recall_curve, auc, normalized_mutual_info_score

if __name__ == '__main__':
    cloudfree = pkl.load(open('pretrained_test_scores_cloudfree.pkl', 'rb'))
    cloudy = pkl.load(open('pretrained_test_scores_cloudy.pkl', 'rb'))
    cloudremoved_glfcr = pkl.load(open('pretrained_test_scores_removed_glf-cr.pkl', 'rb'))
    def plot_conf_mat(data, title):
        # Plot confusion matrix in matplotlib
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(data)
        for (i, j), z in np.ndenumerate(data):
            if z > 0.5:
                color = 'black'
            else:
                color = 'white'
            if z >= 0.01:
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color=color)
            else:
                ax.text(j, i, '0', ha='center', va='center', color=color)

        classes = [IGBPSimpleClasses[i+1] for i in range(10)]
        classes_counts = [IGBPSimpleClasses[i+1] + f" ({cloudfree['clsReport'][str(i+1)]['support']} image(s))" for i in range(10)]
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation='vertical')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes_counts)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    plot_conf_mat(cloudfree['conf_mat'], 'Confusion matrix for cloud-free data')
    plot_conf_mat(cloudy['conf_mat'], 'Confusion matrix for cloudy data')
    plot_conf_mat(cloudremoved_glfcr['conf_mat'], 'Confusion matrix for cloud-removed data with GLF-CR')

    with open("glf-cr_scores.txt") as f:
        glf_cr_scores = f.readlines()

    glf_cr_score_dict = {}
    for line in glf_cr_scores:
        line = line.replace('\n', '').split('\t')
        glf_cr_score_dict[line[0]] = {'psnr': float(line[1]), 'ssim': float(line[2])}

    test_images = cloudfree['id_pred_true_logits'].keys()
    test_images = sorted(test_images)

    info = []
    for i in test_images:
        filename_wo_s2 = i.replace('_s2', '')
        info_dict = {'image': i,
                     'glf-cr_psnr': glf_cr_score_dict[filename_wo_s2]['psnr'],
                     'glf-cr_ssim': glf_cr_score_dict[filename_wo_s2]['ssim'],
                     'true': cloudfree['id_pred_true_logits'][i][1],
                     'pred-cloudfree': cloudfree['id_pred_true_logits'][i][0],
                     'pred-cloudy': cloudy['id_pred_true_logits'][i][0],
                     'pred-glf-cr': cloudremoved_glfcr['id_pred_true_logits'][i][0],
                     'logits-cloudfree': cloudfree['id_pred_true_logits'][i][2],
                     'logits-cloudy': cloudy['id_pred_true_logits'][i][2],
                     'logits-glf-cr': cloudremoved_glfcr['id_pred_true_logits'][i][2]
                     }
        info.append(info_dict)

    df = pd.DataFrame(info)

    df['probs_cloudfree'] = df['logits-cloudfree'].apply(lambda x: np.exp(x) / np.sum(np.exp(x)))
    df['probs_cloudy'] = df['logits-cloudy'].apply(lambda x: np.exp(x) / np.sum(np.exp(x)))
    df['probs_glf-cr'] = df['logits-glf-cr'].apply(lambda x: np.exp(x) / np.sum(np.exp(x)))
    df['logits_sum_cloudfree'] = df['logits-cloudfree'].apply(lambda x: np.sum(x))
    df['logits_sum_cloudy'] = df['logits-cloudy'].apply(lambda x: np.sum(x))
    df['logits_sum_glf-cr'] = df['logits-glf-cr'].apply(lambda x: np.sum(x))
    df['precision_cloudfree'] = df['logits-cloudfree'].apply(lambda x: np.sum(np.exp(x)))
    df['precision_cloudy'] = df['logits-cloudy'].apply(lambda x: np.sum(np.exp(x)))
    df['precision_glf-cr'] = df['logits-glf-cr'].apply(lambda x: np.sum(np.exp(x)))
    df['max_prob_cloudfree'] = df['probs_cloudfree'].apply(lambda x: np.max(x))
    df['max_prob_cloudy'] = df['probs_cloudy'].apply(lambda x: np.max(x))
    df['max_prob_glf-cr'] = df['probs_glf-cr'].apply(lambda x: np.max(x))

    # Replace +inf with max float32 value
    df = df.replace(np.inf, np.finfo(np.float32).max)
    # Replance nan with 0
    df = df.fillna(0)

    def separability_pr(col1, col2):
        # Stack logit sums into one column with corresponding labels
        df1 = df[[col1]]
        df1.columns = ['metric']
        df1['label'] = 1
        df2 = df[[col2]]
        df2.columns = ['metric']
        df2['label'] = 0
        df_sep = pd.concat([df1, df2], axis=0)

        # Precision recall curve
        prec, recall, thresholds = precision_recall_curve(df_sep['label'], df_sep['metric'])
        # Plot precision recall curve
        plt.figure()
        plt.plot(recall, prec)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-recall curve for {col1} and {col2}')
        plt.show()
        auc_val = auc(recall, prec)
        print(f"AUC between {col1} and {col2}: {auc_val}")

    separability_pr('logits_sum_cloudfree', 'logits_sum_cloudy')
    separability_pr('logits_sum_cloudfree', 'logits_sum_glf-cr')
    separability_pr('precision_cloudfree', 'precision_glf-cr')
    separability_pr('precision_cloudfree', 'precision_cloudy')
    separability_pr('max_prob_cloudfree', 'max_prob_cloudy')
    separability_pr('max_prob_cloudfree', 'max_prob_glf-cr')

    pass
