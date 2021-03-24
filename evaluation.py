import sys
import utils
from pyclick.click_models.Evaluation import Perplexity, LogLikelihood

from pyclick.click_models.UBM import UBM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.CTR import GCTR
from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM

CLICK_MODELS = [GCTR(), PBM(), CM(), DBN(), UBM()]


if __name__ == "__main__":
    print("===============================")
    print("The script trains all basic click models and compares their quality.")
    print("===============================")

    if len(sys.argv) < 3:
        print("USAGE: %s <dataset> <sessions_max>" % sys.argv[0])
        print("\tdataset - the path to the dataset from Yandex Relevance Prediction Challenge")
        print("\tsessions_max - the maximum number of one-query search sessions to consider")
        print("")
        sys.exit(1)

    search_sessions_path = sys.argv[1]
    search_sessions_num = int(sys.argv[2])
    train_sessions, train_queries, test_sessions, test_queries = utils.get_train_test(
        search_sessions_path, search_sessions_num
    )

    print("-------------------------------")
    print("Training on %d search sessions (%d unique queries)." % (len(train_sessions), len(train_queries)))
    print("-------------------------------")

    print("-------------------------------")
    print("Testing on %d search sessions (%d unique queries)." % (len(test_sessions), len(test_queries)))
    print("-------------------------------")

    loglikelihood = LogLikelihood()
    perplexity = Perplexity()

    for click_model in CLICK_MODELS:
        click_model.train(train_sessions)

        ll_vallue = loglikelihood.evaluate(click_model, test_sessions)
        perp_values = perplexity.evaluate(click_model, test_sessions)

        print('%s' % click_model.__class__.__name__)
        print('%.6f - log-likelihood' % ll_vallue)
        print('%.6f - perplexity' % perp_values[0])
        print('[%s] - per-rank perplexity' % ', '.join(['%.4f' % perp for perp in perp_values[1]]))
        print('')
