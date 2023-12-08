import json
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

from utils.dataset_utils import *
from sklearn.metrics import precision_recall_curve, auc, f1_score, \
    precision_score, recall_score, roc_curve


def plot_conf_mat(data, title, cloudfree, yticks=True, fontsize=10):
    # Plot confusion matrix in matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(data)
    for (i, j), z in np.ndenumerate(data):
        if z > 0.5:
            color = 'black'
        else:
            color = 'white'
        if z >= 0.01 or z <= -0.01:
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color=color, fontsize=fontsize)
        else:
            ax.text(j, i, '0', ha='center', va='center', color=color, fontsize=fontsize)

    # Get class counts
    classes = [IGBPSimpleClasses[i + 1] for i in range(10)]
    classes_counts = [IGBPSimpleClasses[i + 1] + f" (n:{cloudfree['clsReport'][str(i + 1)]['support']})" for i in
                      range(10)]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation='vertical')
    ax.xaxis.set_ticks_position('bottom')
    if yticks:
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes_counts)
        plt.ylabel('True label')
    else:
        ax.set_yticks([])
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{title}.pdf")
    plt.show()


def plot_bar_metrics(cloudfree, cloudremoved, cloudy, counts, metric):
    # Plot bar chart of metrics
    fig, ax = plt.subplots()
    # Plot all three classes next to each other
    ax.bar(np.arange(0, 10, 1) - 0.2, cloudfree, width=0.2, label='Cloud-free')
    ax.bar(np.arange(0, 10, 1), cloudremoved, width=0.2, label='Cloud-removed')
    ax.bar(np.arange(0, 10, 1) + 0.2, cloudy, width=0.2, label='Cloudy')
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_xticklabels(
        ['0-10% (n={})'.format(counts[0]), '10-20% (n={})'.format(counts[1]), '20-30% (n={})'.format(counts[2]),
         '30-40% (n={})'.format(counts[3]), '40-50% (n={})'.format(counts[4]), '50-60% (n={})'.format(counts[5]),
         '60-70% (n={})'.format(counts[6]), '70-80% (n={})'.format(counts[7]), '80-90% (n={})'.format(counts[8]),
         '90-100% (n={})'.format(counts[9])], rotation=45, ha='right')
    ax.set_xlabel('Cloud cover of co-paired cloudy image')
    ax.set_ylabel(metric)
    # Legend bottom left
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('per_cloud_cover_{}.pdf'.format(metric))
    plt.show()


def separability_pr(df, col1, col2):
    # If col1 or col2 contains cloudy, only choose coverage > 10%
    if 'cloudy' in col1 or 'cloudy' in col2:
        df_sep = df[df['cloud_cover'] > 0.1]
    else:
        df_sep = df.copy()
    # Stack logit sums into one column with corresponding labels
    df1 = df_sep[[col1]].copy()
    df1.columns = ['metric']
    df1['label'] = 1
    df2 = df_sep[[col2]].copy()
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
    print(f"PR AUC between {col1} and {col2}: {auc_val}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(df_sep['label'], df_sep['metric'])
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve for {col1} and {col2}')
    plt.show()
    auc_val = auc(fpr, tpr)
    print(f"ROC AUC between {col1} and {col2}: {auc_val}")


def plot_dist(df, range, field, xlabel, filename, log=False):
    # Plot distribution of field
    plt.figure()
    bins = [i for i in np.arange(range[0], range[1], np.abs(range[1] - range[0]) / 100)]
    kwargs = dict(histtype='stepfilled', alpha=0.4, bins=bins, edgecolor='black')
    plt.hist(df[f'{field}_cloudfree'], label='Cloud-free', **kwargs)
    plt.hist(df[f'{field}_cloudy'], label='Cloudy', **kwargs)
    plt.hist(df[f'{field}_cloudremoved'], label='Cloud-removed', **kwargs)
    plt.xlabel(xlabel)
    plt.xlim(range[0], range[1])
    if log:
        plt.yscale('log')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def create_all_info_dataframe():
    # Create dataframe with all information for each image
    with open('ResultScores/cloud_cover.txt') as f:
        cloud_cover = f.readlines()
    with open("ResultScores/uncrtaints_scores.json") as f:
        uncrtaints_scores = json.load(f)

    cloud_cover_dict = {}
    for line in cloud_cover:
        line = line.replace('\n', '').split('\t')
        cloud_cover_dict[line[0]] = float(line[1])
    uncrtaints_scores_dict = {}
    for entry in uncrtaints_scores:
        uncrtaints_scores_dict[entry['file_name']] = entry

    test_images = cloudfree['id_pred_true_logits'].keys()
    test_images = sorted(test_images)

    info = []
    info_dict = {}
    for i in test_images:
        filename_wo_s2 = i.replace('_s2', '')
        logits_cloudfree = cloudfree['id_pred_true_logits'][i][2]
        logits_cloudy = cloudy['id_pred_true_logits'][i][2]
        logits_cloudremoved = cloudremoved['id_pred_true_logits'][i][2]
        probs_cloudfree = np.exp(logits_cloudfree) / (np.sum(np.exp(logits_cloudfree)) + 1e-15)
        probs_cloudy = np.exp(logits_cloudy) / (np.sum(np.exp(logits_cloudy)) + 1e-15)
        probs_cloudremoved = np.exp(logits_cloudremoved) / (np.sum(np.exp(logits_cloudremoved)) + 1e-15)
        image_info_dict = {'image': i,
                           'cloud_cover': cloud_cover_dict[i],
                           'uncrtaints_psnr': uncrtaints_scores_dict[filename_wo_s2]['PSNR'],
                           'uncrtaints_ssim': uncrtaints_scores_dict[filename_wo_s2]['SSIM'],
                           'true': cloudfree['id_pred_true_logits'][i][1],
                           'pred-cloudfree': cloudfree['id_pred_true_logits'][i][0],
                           'pred-cloudy': cloudy['id_pred_true_logits'][i][0],
                           'pred-cloudremoved': cloudremoved['id_pred_true_logits'][i][0],
                           'logits-cloudfree': logits_cloudfree,
                           'logits-cloudy': logits_cloudy,
                           'logits-cloudremoved': logits_cloudremoved,
                           'probs_cloudfree': probs_cloudfree,
                           'probs_cloudy': probs_cloudy,
                           'probs_cloudremoved': probs_cloudremoved,
                           'max_prob_cloudfree': np.max(probs_cloudfree),
                           'max_prob_cloudy': np.max(probs_cloudy),
                           'max_prob_cloudremoved': np.max(probs_cloudremoved),
                           }
        info.append(image_info_dict)
        info_dict[i] = image_info_dict

    # Save dataframe and dict
    df = pd.DataFrame(info)
    pkl.dump(df, open('ResultScores/test_set_all_info.pkl', 'wb'))
    pkl.dump(info_dict, open('ResultScores/test_set_all_info_dict.pkl', 'wb'))


if __name__ == '__main__':

    # Load test scores
    cloudfree = pkl.load(open('ResultScores/test_scores_cloudfree.pkl', 'rb'))
    cloudy = pkl.load(open('ResultScores/test_scores_cloudy.pkl', 'rb'))
    cloudremoved = pkl.load(open('ResultScores/test_scores_cloudremoved_uncrtaints.pkl', 'rb'))
    val_counts = pkl.load(open('ResultScores/val_counts.pkl', 'rb'))
    train_counts = pkl.load(open('ResultScores/train_counts.pkl', 'rb'))

    if not os.path.exists('ResultScores/test_set_all_info.pkl'):
        create_all_info_dataframe()
    df = pkl.load(open('ResultScores/test_set_all_info.pkl', 'rb'))


    #############################
    # Plot confusion matrices
    #############################
    plot_conf_mat(cloudfree['conf_mat'], 'Confusion matrix for cloud-free data', cloudfree)
    plot_conf_mat(cloudy['conf_mat'], 'Confusion matrix for cloudy data', cloudfree, yticks=False)
    plot_conf_mat(cloudremoved['conf_mat'], 'Confusion matrix for cloud-removed data', cloudfree, yticks=False)
    # Difference between cloudfree and cloudremoved confusion matrices
    plot_conf_mat(cloudremoved['conf_mat'] - cloudfree['conf_mat'], 'Difference cloud-free and -removed', cloudfree, yticks=True, fontsize=8)

    #############################
    # Plot histogram of cloud cover
    #############################
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
    plt.tight_layout()
    plt.savefig('cloud_cover_hist.pdf')
    plt.show()

    #############################
    # Plot histogram of true label counts
    #############################
    fig, ax = plt.subplots()
    test_counts = df['true'].value_counts().sort_index()
    # Fill in missing labels
    test_counts = np.array([0 if i not in test_counts.index else test_counts[i] for i in np.arange(0, 10, 1)])
    # Normalize
    test_counts = test_counts / np.sum(test_counts)
    train_counts = train_counts / np.sum(train_counts)
    val_counts = val_counts / np.sum(val_counts)
    # Plot all three sets
    ax.bar(np.arange(0, 10, 1) - 0.2, train_counts, width=0.2, label='Train')
    ax.bar(np.arange(0, 10, 1), val_counts, width=0.2, label='Val')
    ax.bar(np.arange(0, 10, 1) + 0.2, test_counts, width=0.2, label='Test')
    ax.set_xticks([i for i in np.arange(0, 10, 1)])
    ax.set_xticklabels(
        IGBPSimpleClassList, rotation=45, ha='right')
    ax.set_xlabel('True label')
    ax.set_ylabel('Percentage of images')
    plt.tight_layout()
    plt.legend()
    plt.savefig('true_label_hist.pdf')
    plt.show()

    #############################
    # Compute metrics grouped by cloud cover
    #############################
    df['cloud_cover_bin'] = pd.cut(df['cloud_cover'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                   labels=False)
    # Group by cloud cover bin
    df_grouped = df.groupby('cloud_cover_bin')
    # Get counts
    test_counts = df_grouped.apply(lambda x: len(x))
    # Compute scores for each group
    f1_cloudremoved = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-cloudremoved'], average='macro'))
    f1_cloudy = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-cloudy'], average='macro'))
    f1_cloudfree = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-cloudfree'], average='macro'))
    precision_cloudremoved = df_grouped.apply(lambda x: precision_score(x['true'], x['pred-cloudremoved'], average='macro'))
    precision_cloudy = df_grouped.apply(lambda x: precision_score(x['true'], x['pred-cloudy'], average='macro'))
    precision_cloudfree = df_grouped.apply(lambda x: precision_score(x['true'], x['pred-cloudfree'], average='macro'))
    recall_cloudremoved = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-cloudremoved'], average='macro'))
    recall_cloudy = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-cloudy'], average='macro'))
    recall_cloudfree = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-cloudfree'], average='macro'))
    avg_prob_cloudremoved = df_grouped.apply(lambda x: np.mean(x['max_prob_cloudremoved']))
    avg_prob_cloudy = df_grouped.apply(lambda x: np.mean(x['max_prob_cloudy']))
    avg_prob_cloudfree = df_grouped.apply(lambda x: np.mean(x['max_prob_cloudfree']))

    #############################
    # Plot metrics grouped by cloud cover
    #############################
    plot_bar_metrics(f1_cloudfree, f1_cloudremoved, f1_cloudy, test_counts, 'F1 score')
    plot_bar_metrics(precision_cloudfree, precision_cloudremoved, precision_cloudy, test_counts, 'Precision')
    plot_bar_metrics(recall_cloudfree, recall_cloudremoved, recall_cloudy, test_counts, 'Recall')
    plot_bar_metrics(avg_prob_cloudfree, avg_prob_cloudremoved, avg_prob_cloudy, test_counts, 'Average probability')

    #############################
    # Compute metrics grouped by SSIM
    #############################
    df['ssim_bin'] = pd.cut(df['uncrtaints_ssim'], bins=[0.7, 0.8, 0.9, 1], labels=False)
    # Group by SSIM bin
    df_grouped = df.groupby('ssim_bin')
    # Counts for each group
    test_counts = df_grouped.apply(lambda x: len(x))
    # Compute scores for each group
    f1_cloudremoved_ssim = df_grouped.apply(lambda x: f1_score(x['true'], x['pred-cloudremoved'], average='macro'))
    precision_cloudremoved_ssim = df_grouped.apply(lambda x: precision_score(x['true'], x['pred-cloudremoved'], average='macro'))
    recall_cloudremoved_ssim = df_grouped.apply(lambda x: recall_score(x['true'], x['pred-cloudremoved'], average='macro'))
    avg_prob_cloudremoved_ssim = df_grouped.apply(lambda x: np.mean(x['max_prob_cloudremoved']))

    #############################
    # Bar plot metrics grouped by SSIM
    #############################
    fig, ax = plt.subplots()
    # Plot f1 cloud-removed for each SSIM bin next to each other
    ax.bar(np.arange(0, 3, 1) + 0.075, f1_cloudremoved_ssim, width=0.15, label='F1-score')
    ax.bar(np.arange(0, 3, 1) - 0.225, precision_cloudremoved_ssim, width=0.15, label='Precision')
    ax.bar(np.arange(0, 3, 1) - 0.075, recall_cloudremoved_ssim, width=0.15, label='Recall')
    ax.bar(np.arange(0, 3, 1) + 0.225, avg_prob_cloudremoved_ssim, width=0.15, label='Average probability')
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels(
        ['0.7-0.8 (n={})'.format(test_counts[0]), '0.8-0.9 (n={})'.format(test_counts[1]), '0.9-1.0 (n={})'.format(test_counts[2])])

    ax.set_xlabel('SSIM between cloud-removed and cloud-free image')
    ax.set_ylabel('Metric')
    plt.tight_layout()
    plt.legend()
    plt.savefig('per_ssim.pdf')
    plt.show()


    #############################
    # SEPARABILITY STUDY
    #############################
    def calc_entropy(row, col):
        one_hot = np.zeros(10)
        one_hot[row['true']] = 1
        return -np.sum(np.log(row[f'probs_{col}']) * one_hot)

    df['logits_sum_cloudfree'] = df['logits-cloudfree'].apply(lambda x: np.sum(x))
    df['logits_sum_cloudy'] = df['logits-cloudy'].apply(lambda x: np.sum(x))
    df['logits_sum_cloudremoved'] = df['logits-cloudremoved'].apply(lambda x: np.sum(x))
    df['precision_cloudfree'] = df['logits-cloudfree'].apply(lambda x: np.sum(np.exp(x)))
    df['precision_cloudy'] = df['logits-cloudy'].apply(lambda x: np.sum(np.exp(x)))
    df['precision_cloudremoved'] = df['logits-cloudremoved'].apply(lambda x: np.sum(np.exp(x)))
    df['cross_entropy_cloudfree'] = df.apply(calc_entropy, col='cloudfree', axis=1)
    df['cross_entropy_cloudy'] = df.apply(calc_entropy, col='cloudy', axis=1)
    df['cross_entropy_cloudremoved'] = df.apply(calc_entropy, col='cloudremoved', axis=1)

    # Replace +inf with max float32 value
    df = df.replace(np.inf, np.finfo(np.float32).max)
    # Replance nan with 0
    df = df.fillna(0)

    # Calculate average of max probabilities
    print(f"Average max probability of cloudfree: {np.mean(df['max_prob_cloudfree'])}")
    print(f"Average max probability of cloudy: {np.mean(df['max_prob_cloudy'])}")
    print(f"Average max probability of cloudremoved: {np.mean(df['max_prob_cloudremoved'])}")

    separability_pr(df, 'logits_sum_cloudfree', 'logits_sum_cloudy')
    separability_pr(df, 'logits_sum_cloudfree', 'logits_sum_cloudremoved')
    separability_pr(df, 'precision_cloudfree', 'precision_cloudy')
    separability_pr(df, 'precision_cloudfree', 'precision_cloudremoved')
    separability_pr(df, 'max_prob_cloudfree', 'max_prob_cloudy')
    separability_pr(df, 'max_prob_cloudfree', 'max_prob_cloudremoved')
    separability_pr(df, 'cross_entropy_cloudfree', 'cross_entropy_cloudy')
    separability_pr(df, 'cross_entropy_cloudfree', 'cross_entropy_cloudremoved')

    #############################
    # Plot distributions
    #############################
    plot_dist(df, (0, 2), 'cross_entropy', 'Cross entropy', 'dist_cross_entropy.pdf', log=True)
    plot_dist(df, (0, 0.5), 'cross_entropy', 'Cross entropy', 'dist_cross_entropy_zoom.pdf', log=True)
    plot_dist(df, (0, 700), 'precision', 'Precision', 'dist_precision.pdf')
    plot_dist(df, (-50, -5), 'logits_sum', 'Logits sum', 'dist_logits_sum.pdf')
    plot_dist(df, (0.5, 1), 'max_prob', 'Max probability', 'dist_max_prob.pdf')

