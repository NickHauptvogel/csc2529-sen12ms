import pickle as pkl

if __name__ == '__main__':
    cloudfree = pkl.load(open('test_scores_cloudfree.pkl', 'rb'))
    cloudy = pkl.load(open('test_scores_cloudy.pkl', 'rb'))
    cloudremoved = pkl.load(open('test_scores_cloudremoved.pkl', 'rb'))

    pass
