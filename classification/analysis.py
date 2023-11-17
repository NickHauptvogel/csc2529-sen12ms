import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_utils import *
from sklearn.metrics import precision_recall_curve, auc, normalized_mutual_info_score, f1_score, \
    precision_score, recall_score


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

    classes = [IGBPSimpleClasses[i + 1] for i in range(10)]
    classes_counts = [IGBPSimpleClasses[i + 1] + f" ({cloudfree['clsReport'][str(i + 1)]['support']} image(s))" for i in
                      range(10)]
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


cloudfree = pkl.load(open('test_scores_cloudfree.pkl', 'rb'))
cloudy = pkl.load(open('test_scores_cloudy.pkl', 'rb'))
cloudremoved_glfcr = pkl.load(open('test_scores_removed_glf-cr.pkl', 'rb'))


def create_all_info_dataframe():
    with open('cloud_cover.txt') as f:
        cloud_cover = f.readlines()
    with open("glf-cr_scores.txt") as f:
        glf_cr_scores = f.readlines()
    cloud_cover_dict = {}
    for line in cloud_cover:
        line = line.replace('\n', '').split('\t')
        cloud_cover_dict[line[0]] = float(line[1])
    glf_cr_score_dict = {}
    for line in glf_cr_scores:
        line = line.replace('\n', '').split('\t')
        glf_cr_score_dict[line[0]] = {'psnr': float(line[1]), 'ssim': float(line[2])}

    test_images = cloudfree['id_pred_true_logits'].keys()
    test_images = sorted(test_images)

    info = []
    info_dict = {}
    for i in test_images:
        filename_wo_s2 = i.replace('_s2', '')
        logits_cloudfree = cloudfree['id_pred_true_logits'][i][2]
        logits_cloudy = cloudy['id_pred_true_logits'][i][2]
        logits_glf_cr = cloudremoved_glfcr['id_pred_true_logits'][i][2]
        probs_cloudfree = np.exp(logits_cloudfree) / (np.sum(np.exp(logits_cloudfree)) + 1e-15)
        probs_cloudy = np.exp(logits_cloudy) / (np.sum(np.exp(logits_cloudy)) + 1e-15)
        probs_glf_cr = np.exp(logits_glf_cr) / (np.sum(np.exp(logits_glf_cr)) + 1e-15)
        image_info_dict = {'image': i,
                           'cloud_cover': cloud_cover_dict[i],
                           'glf-cr_psnr': glf_cr_score_dict[filename_wo_s2]['psnr'],
                           'glf-cr_ssim': glf_cr_score_dict[filename_wo_s2]['ssim'],
                           'true': cloudfree['id_pred_true_logits'][i][1],
                           'pred-cloudfree': cloudfree['id_pred_true_logits'][i][0],
                           'pred-cloudy': cloudy['id_pred_true_logits'][i][0],
                           'pred-glf-cr': cloudremoved_glfcr['id_pred_true_logits'][i][0],
                           'logits-cloudfree': logits_cloudfree,
                           'logits-cloudy': logits_cloudy,
                           'logits-glf-cr': logits_glf_cr,
                           'probs_cloudfree': probs_cloudfree,
                           'probs_cloudy': probs_cloudy,
                           'probs_glf-cr': probs_glf_cr,
                           'max_prob_cloudfree': np.max(probs_cloudfree),
                           'max_prob_cloudy': np.max(probs_cloudy),
                           'max_prob_glf-cr': np.max(probs_glf_cr),
                           }
        info.append(image_info_dict)
        info_dict[i] = image_info_dict

    df = pd.DataFrame(info)
    pkl.dump(df, open('test_set_all_info.pkl', 'wb'))
    pkl.dump(info_dict, open('test_set_all_info_dict.pkl', 'wb'))


if __name__ == '__main__':

    if not os.path.exists('test_set_all_info.pkl'):
        create_all_info_dataframe()
    df = pkl.load(open('test_set_all_info.pkl', 'rb'))

    plot_conf_mat(cloudfree['conf_mat'], 'Confusion matrix for cloud-free data')
    plot_conf_mat(cloudy['conf_mat'], 'Confusion matrix for cloudy data')
    plot_conf_mat(cloudremoved_glfcr['conf_mat'], 'Confusion matrix for cloud-removed data with GLF-CR')

    # Histogram of cloud cover
    fig, ax = plt.subplots()
    ax.hist(df['cloud_cover'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], edgecolor='white',
            linewidth=2)
    # x ticks are bin edges, so we need to shift them by half a bin width
    ax.set_xticks([i + 0.05 for i in np.arange(0, 1.0, 0.1)])
    ax.set_xticklabels(
        ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
        rotation=45, ha='right')
    ax.set_xlabel('Cloud cover')
    ax.set_ylabel('Number of images')
    plt.show()

    # Compute metrics grouped by cloud cover
    df['cloud_cover_bin'] = pd.cut(df['cloud_cover'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                   labels=False)
    # Group by cloud cover bin
    df_grouped = df.groupby('cloud_cover_bin')
    # Get counts
    counts = df_grouped.apply(lambda x: len(x))
    # Compute scores for each group
    f1_glf_cr = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-glf-cr'], average='weighted'))
    f1_cloudy = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-cloudy'], average='weighted'))
    f1_cloudfree = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-cloudfree'], average='weighted'))
    precision_glf_cr = df_grouped.apply(lambda x: precision_score(x['true'], x['pred-glf-cr'], average='weighted'))
    precision_cloudy = df_grouped.apply(lambda x: precision_score(x['true'], x['pred-cloudy'], average='weighted'))
    precision_cloudfree = df_grouped.apply(
        lambda x: precision_score(x['true'], x['pred-cloudfree'], average='weighted'))
    recall_glf_cr = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-glf-cr'], average='weighted'))
    recall_cloudy = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-cloudy'], average='weighted'))
    recall_cloudfree = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-cloudfree'], average='weighted'))
    avg_prob_glf_cr = df_grouped.apply(lambda x: np.mean(x['max_prob_glf-cr']))
    avg_prob_cloudy = df_grouped.apply(lambda x: np.mean(x['max_prob_cloudy']))
    avg_prob_cloudfree = df_grouped.apply(lambda x: np.mean(x['max_prob_cloudfree']))


    def plot_bar_metrics(cloudfree, glf_cr, cloudy, counts, metric):
        fig, ax = plt.subplots()
        ax.bar(np.arange(0, 10, 1) - 0.2, cloudfree, width=0.2, label='Cloud-free')
        ax.bar(np.arange(0, 10, 1), glf_cr, width=0.2, label='GLF-CR')
        ax.bar(np.arange(0, 10, 1) + 0.2, cloudy, width=0.2, label='Cloudy')
        ax.set_xticks(np.arange(0, 10, 1))
        ax.set_xticklabels(
            ['0-10% (n={})'.format(counts[0]), '10-20% (n={})'.format(counts[1]), '20-30% (n={})'.format(counts[2]),
             '30-40% (n={})'.format(counts[3]), '40-50% (n={})'.format(counts[4]), '50-60% (n={})'.format(counts[5]),
             '60-70% (n={})'.format(counts[6]), '70-80% (n={})'.format(counts[7]), '80-90% (n={})'.format(counts[8]),
             '90-100% (n={})'.format(counts[9])], rotation=45, ha='right')
        ax.set_xlabel('Cloud cover of co-paired cloudy image')
        ax.set_ylabel(metric)
        ax.legend()
        plt.tight_layout()
        plt.show()


    plot_bar_metrics(f1_cloudfree, f1_glf_cr, f1_cloudy, counts, 'F1 score')
    plot_bar_metrics(precision_cloudfree, precision_glf_cr, precision_cloudy, counts, 'Precision')
    plot_bar_metrics(recall_cloudfree, recall_glf_cr, recall_cloudy, counts, 'Recall')
    plot_bar_metrics(avg_prob_cloudfree, avg_prob_glf_cr, avg_prob_cloudy, counts, 'Average probability')

    # Compute metrics grouped by SSIM
    df['ssim_bin'] = pd.cut(df['glf-cr_ssim'], bins=[0.7, 0.8, 0.9, 1], labels=False)
    # Group by SSIM bin
    df_grouped = df.groupby('ssim_bin')
    # Counts for each group
    counts = df_grouped.apply(lambda x: len(x))
    # Compute scores for each group
    f1_glf_cr_ssim = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-glf-cr'], average='weighted'))
    precision_glf_cr_ssim = df_grouped.apply(lambda x: precision_score(x['true'], x['pred-glf-cr'], average='weighted'))
    recall_glf_cr_ssim = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-glf-cr'], average='weighted'))
    avg_prob_glf_cr_ssim = df_grouped.apply(lambda x: np.mean(x['max_prob_glf-cr']))


    def plot_bar_metrics_ssim(glf_cr, counts, metric):
        fig, ax = plt.subplots()
        ax.bar(np.arange(0, 3, 1), glf_cr, label='GLF-CR')
        ax.set_xticks(np.arange(0, 3, 1))
        ax.set_xticklabels(
            ['70-80% (n={})'.format(counts[0]), '80-90% (n={})'.format(counts[1]), '90-100% (n={})'.format(counts[2])],
            rotation=45, ha='right')
        ax.set_xlabel('SSIM of GLF-CR between cloud-removed and cloud-free image')
        ax.set_ylabel(metric)
        plt.tight_layout()
        plt.show()


    plot_bar_metrics_ssim(f1_glf_cr_ssim, counts, 'F1 score')
    plot_bar_metrics_ssim(precision_glf_cr_ssim, counts, 'Precision')
    plot_bar_metrics_ssim(recall_glf_cr_ssim, counts, 'Recall')
    plot_bar_metrics_ssim(avg_prob_glf_cr_ssim, counts, 'Average probability')

    df['logits_sum_cloudfree'] = df['logits-cloudfree'].apply(lambda x: np.sum(x))
    df['logits_sum_cloudy'] = df['logits-cloudy'].apply(lambda x: np.sum(x))
    df['logits_sum_glf-cr'] = df['logits-glf-cr'].apply(lambda x: np.sum(x))
    df['precision_cloudfree'] = df['logits-cloudfree'].apply(lambda x: np.sum(np.exp(x)))
    df['precision_cloudy'] = df['logits-cloudy'].apply(lambda x: np.sum(np.exp(x)))
    df['precision_glf-cr'] = df['logits-glf-cr'].apply(lambda x: np.sum(np.exp(x)))

    # Replace +inf with max float32 value
    df = df.replace(np.inf, np.finfo(np.float32).max)
    # Replance nan with 0
    df = df.fillna(0)

    # Calculate average of max probabilities
    print(f"Average max probability of cloudfree: {np.mean(df['max_prob_cloudfree'])}")
    print(f"Average max probability of cloudy: {np.mean(df['max_prob_cloudy'])}")
    print(f"Average max probability of glf-cr: {np.mean(df['max_prob_glf-cr'])}")


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
