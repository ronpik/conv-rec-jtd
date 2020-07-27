#include "common.hpp"
#include "vector"
#include "map"
#include "string"
#include "limits"
#include "math.h"
#include "omp.h"
#include "lbfgs.h"
#include "sys/time.h"
#include "language.hpp"

#define BACK 0
#define DISC 1
#define TOPIC 2
#define RRATING 1
using namespace std;


inline double square(double x) {
    return x * x;
}

inline double dsquare(double x) {
    return 2 * x;
}

/*
double clock_()
{
  timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec + (tim.tv_usec / 1000000.0);
}
*/

/// Recover all parameters from a vector (g)
// W, &alpha, &kappa_disc, &kappa_topic, &beta_user, &beta_conv, &gamma_user, &gamma_conv, &delta_user, &delta_conv, &topo_user, &topicWords, &discourseWords, &backgroundWords, &typeSwitcher, true
int topicCorpus::getG(double *g,
                      double **alpha,
                      double **kappa_disc,
                      double **kappa_topic,
                      double **beta_user,
                      double **beta_conv,
                      double ***gamma_user,
                      double ***gamma_conv,
                      double ***delta_user,
                      double ***delta_conv,
                      double ***topo_user,
                      double ***topicWords,
                      double ***discourseWords,
                      double **backgroundWords,
                      double ***typeSwitcher,
                      bool init) {
    if (init) {
        *gamma_user = new double *[nUsers];
        *gamma_conv = new double *[nConvs];
        *delta_user = new double *[nUsers];
        *delta_conv = new double *[nConvs];
        *topo_user = new double *[nUsers];
        *topicWords = new double *[nWords];
        *discourseWords = new double *[nWords];
        *typeSwitcher = new double *[D];
    }

    int ind = 0;
    *alpha = g + ind;
    ind++;
    *kappa_disc = g + ind;
    ind++;
    *kappa_topic = g + ind;
    ind++;

    *beta_user = g + ind;
    ind += nUsers;
    *beta_conv = g + ind;
    ind += nConvs;

    for (int u = 0; u < nUsers; u++) {
        (*gamma_user)[u] = g + ind;
        ind += K;
    }
    for (int b = 0; b < nConvs; b++) {
        (*gamma_conv)[b] = g + ind;
        ind += K;
    }
    for (int u = 0; u < nUsers; u++) {
        (*delta_user)[u] = g + ind;
        ind += D;
    }
    for (int b = 0; b < nConvs; b++) {
        (*delta_conv)[b] = g + ind;
        ind += D;
    }
    for (int u = 0; u < nUsers; u++) {
        (*topo_user)[u] = g + ind;
        ind += embeddingSize;
    }
    for (int w = 0; w < nWords; w++) {
        (*topicWords)[w] = g + ind;
        ind += K;
    }
    for (int w = 0; w < nWords; w++) {
        (*discourseWords)[w] = g + ind;
        ind += D;
    }
    *backgroundWords = g + ind;
    ind += nWords;
    for (int d = 0; d < D; d++) {
        (*typeSwitcher)[d] = g + ind;
        ind += 3;
    }

    if (ind != NW) {
        printf("Got incorrect index at line %d\n", __LINE__);
        exit(1);
    }
    return ind;
}

/// Free parameters
void topicCorpus::clearG(double **alpha,
                         double **kappa_disc,
                         double **kappa_topic,
                         double **beta_user,
                         double **beta_conv,
                         double ***gamma_user,
                         double ***gamma_conv,
                         double ***delta_user,
                         double ***delta_conv,
                         double ***topo_user,
                         double ***topicWords,
                         double ***discourseWords,
                         double **backgroundWords,
                         double ***typeSwitcher) {
    delete[] (*gamma_user);
    delete[] (*gamma_conv);
    delete[] (*delta_user);
    delete[] (*delta_conv);
    delete[] (*topo_user);
    delete[] (*topicWords);
    delete[] (*discourseWords);
    delete[] (*typeSwitcher);
}

/// Compute energy
static lbfgsfloatval_t evaluate(void *instance,
                                const lbfgsfloatval_t *x,
                                lbfgsfloatval_t *g,
                                const int n,
                                const lbfgsfloatval_t step) {
    topicCorpus *ec = (topicCorpus *) instance;

    for (int i = 0; i < ec->NW; i++)
        ec->W[i] = x[i];

    double *grad = new double[ec->NW];
    ec->dl(grad);
    for (int i = 0; i < ec->NW; i++)
        g[i] = grad[i];
    delete[] grad;

    lbfgsfloatval_t fx = ec->lsq();
    return fx;
}

static int progress(void *instance,
                    const lbfgsfloatval_t *x,
                    const lbfgsfloatval_t *g,
                    const lbfgsfloatval_t fx,
                    const lbfgsfloatval_t xnorm,
                    const lbfgsfloatval_t gnorm,
                    const lbfgsfloatval_t step,
                    int n,
                    int k,
                    int ls) {
    // static double gtime = clock_();
    printf(".");
    fflush(stdout);
    // double tdiff = clock_();
    // gtime = tdiff;
    return 0;
}

/// Predict a particular rating given the current parameter values
double topicCorpus::prediction(int user, int conv) {
    double res = *alpha + beta_user[user] + beta_conv[conv];
    for (int k = 0; k < K; k++)
        res += topicProportion * gamma_user[user][k] * gamma_conv[conv][k];
    for (int d = 0; d < D; d++)
        res += (1 - topicProportion) * delta_user[user][d] * delta_conv[conv][d];
    for (int w = 0; w < embeddingSize; w++)
        res += topo_user[user][w] * conversationEmbeddings[conv][w];
    return res;
}

/// Compute normalization constant for a particular item
void topicCorpus::topicZ(int conv, double &res) {
    res = 0;
    for (int k = 0; k < K; k++)
        res += exp(*kappa_topic * gamma_conv[conv][k]);
}

/// Compute normalization constant for a particular user
void topicCorpus::discourseZ(int user, double &res) {
    res = 0;
    for (int d = 0; d < D; d++)
        res += exp(*kappa_disc * delta_user[user][d]);
}

/// Compute normalization constants for all K topics
void topicCorpus::wordtopicZ(double *res) {
    for (int k = 0; k < K; k++) {
        res[k] = 0;
        for (int w = 0; w < nWords; w++)
            res[k] += exp(bg_topicWords[w] + topicWords[w][k]);
    }
}

/// Compute normalization constants for all D discourses
void topicCorpus::worddiscourseZ(double *res) {
    for (int d = 0; d < D; d++) {
        res[d] = 0;
        for (int w = 0; w < nWords; w++)
            res[d] += exp(bg_discourseWords[w] + discourseWords[w][d]);
    }
}

/// Compute normalization constants for backgroundwords
void topicCorpus::wordbackgroundZ(double &res) {
    res = 0;
    for (int w = 0; w < nWords; w++)
        res += exp(backgroundWords[w]);
}

/// Compute normalization constants for all discourses of 3 wordtypes
void topicCorpus::wordtypeZ(double *res) {
    for (int d = 0; d < D; d++) {
        res[d] = 0;
        for (int t = 0; t < 3; t++)
            res[d] += exp(typeSwitcher[d][t]);
    }
}

/// Sampling an assignment with scores(i.e. prob. after normalizing) and num(number of dimensions)
int sampleScores(double *scores, int num) {
    double total = 0;
    for (int n = 0; n < num; n++) {
        total += scores[n];
    }
    for (int n = 0; n < num; n++)
        scores[n] /= total;
    int newSample = 0;
    double x = rand() * 1.0 / (1.0 + RAND_MAX);
    while (true) {
        x -= scores[newSample];
        if (x < 0)
            break;
        newSample++;
    }
    return newSample;
}

/// Update topic and discourse assignments for each message and word
void topicCorpus::msgSampling() {
    // double updateStart = clock_();
    // printf("%f %f %f\n", typeSwitcher[0][0], typeSwitcher[0][1], typeSwitcher[0][2]);
    for (int c = 0; c < nConvs; c++) {
        if (c > 0 and c % 10000 == 0) {
            printf(".");
            fflush(stdout);
        }

        for (int x = 0; x < (int) trainVotesPerConv[c].size(); x++) {
            vote *vi = trainVotesPerConv[c][x];
            int cv = vi->item;
            int ur = vi->user;

            // Sampling topic for this message
            double *topicScores = new double[K];
            for (int k = 0; k < K; k++)
                topicScores[k] = exp(*kappa_topic * gamma_conv[cv][k]);
            int newTopic = sampleScores(topicScores, K);
            delete[] topicScores;
            int oldTopic = msgTopics[vi];
            if (newTopic != oldTopic) { // Update topic counts if the topic for this message changed
                convTopicSums[cv]--;
                convTopicSums[cv]++;
                convTopicCounts[cv][oldTopic]--;
                convTopicCounts[cv][newTopic]++;
                msgTopics[vi] = newTopic;
            }

            // Sampling discourse for this message
            double *discourseScores = new double[D];
            for (int d = 0; d < D; d++)
                discourseScores[d] = exp(*kappa_disc * delta_user[ur][d]);
            int newDiscourse = sampleScores(discourseScores, D);
            delete[] discourseScores;
            int oldDiscourse = msgDiscourses[vi];
            if (newDiscourse != oldDiscourse) { // Update discourse counts if the discourse for this message changed
                userDiscourseSums[ur]--;
                userDiscourseSums[ur]++;
                userDiscourseCounts[ur][oldDiscourse]--;
                userDiscourseCounts[ur][newDiscourse]++;
                msgDiscourses[vi] = newDiscourse;
            }

            // Sample word type at word level
            for (int wp = 0; wp < (int) vi->words.size(); wp++) {
                int wi = vi->words[wp]; // The word
                double *typeScores = new double[3];
                typeScores[0] = exp(typeSwitcher[newDiscourse][0] + backgroundWords[wi]);
                typeScores[1] = exp(
                        typeSwitcher[newDiscourse][1] + bg_discourseWords[wi] + discourseWords[wi][newDiscourse]);
                if (std::find(stopWords.begin(), stopWords.end(), wi) != stopWords.end())
                    typeScores[2] = 0;
                else
                    typeScores[2] = exp(typeSwitcher[newDiscourse][2] + bg_topicWords[wi] + topicWords[wi][newTopic]);
                int newType = sampleScores(typeScores, 3);
                delete[] typeScores;
                int oldType = msgWordtypes[vi][wp];

                // Update assignment for the word
                if (oldType != newType || newDiscourse != oldDiscourse) {
                    discourseWordtypeSums[oldDiscourse]--;
                    discourseWordtypeSums[newDiscourse]++;
                    discourseWordtypeCounts[oldDiscourse][oldType]--;
                    discourseWordtypeCounts[newDiscourse][newType]++;
                    msgWordtypes[vi][wp] = newType;
                }
                if (oldType == BACK) {
                    backgroundCounts -= 1;
                    wordBackgroundCounts[wi] -= 1;
                } else if (oldType == DISC) {
                    wordDiscourseSums[oldDiscourse] -= 1;
                    wordDiscourseCounts[wi][oldDiscourse] -= 1;
                } else // oldType == Topic
                {
                    wordTopicSums[oldTopic] -= 1;
                    wordTopicCounts[wi][oldTopic] -= 1;
                }

                if (newType == BACK) {
                    backgroundCounts += 1;
                    wordBackgroundCounts[wi] += 1;
                } else if (newType == DISC) {
                    wordDiscourseSums[newDiscourse] += 1;
                    wordDiscourseCounts[wi][newDiscourse] += 1;
                } else // newType == Topic
                {
                    wordTopicSums[newTopic] += 1;
                    wordTopicCounts[wi][newTopic] += 1;
                }
            }
        }
    }
    printf("\n");
}

/// Subtract averages from word weights so that each word has average weight zero across all topics, discourses (the remaining weight is stored in "bg")
void topicCorpus::normalizeWordWeights(void) {
    for (int w = 0; w < nWords; w++) {
        double av = 0;
        for (int k = 0; k < K; k++)
            av += topicWords[w][k];
        av /= K;
        for (int k = 0; k < K; k++)
            topicWords[w][k] -= av;
        bg_topicWords[w] += av;
        av = 0;
        for (int d = 0; d < D; d++)
            av += discourseWords[w][d];
        av /= D;
        for (int d = 0; d < D; d++)
            discourseWords[w][d] -= av;
        bg_discourseWords[w] += av;
    }
}

/// Derivative of the energy function, cost function is MSE
void topicCorpus::dl(double *grad) {
    // double dlStart = clock_();

    for (int w = 0; w < NW; w++)
        grad[w] = 0;

    double *dalpha;
    double *dkappa_disc;
    double *dkappa_topic;
    double *dbeta_user;
    double *dbeta_conv;
    double **dgamma_user;
    double **dgamma_conv;
    double **ddelta_user;
    double **ddelta_conv;
    double **dtopo_user;
    double **dtopicWords;
    double **ddiscourseWords;
    double *dbackgroundWords;
    double **dtypeSwitcher;

    getG(grad, &(dalpha), &(dkappa_disc), &(dkappa_topic), &(dbeta_user), &(dbeta_conv), &(dgamma_user), &(dgamma_conv),
         &(ddelta_user), &(ddelta_conv), &(dtopo_user), &(dtopicWords), &(ddiscourseWords), &(dbackgroundWords), &(dtypeSwitcher),
         true);

    double da = 0;
#pragma omp parallel for reduction(+:da)
    for (int u = 0; u < nUsers; u++) {
        for (vector<vote *>::iterator it = trainVotesPerUser[u].begin(); it != trainVotesPerUser[u].end(); it++) {
            vote *vi = *it;
            double p1 = prediction(u, vi->item);
            double dl1 = dsquare(p1 - RRATING) * confidence[u][vi->item];

            da += dl1;
            dbeta_user[u] += dl1;
            for (int k = 0; k < K; k++)
                dgamma_user[u][k] += dl1 * topicProportion * gamma_conv[vi->item][k];
            for (int d = 0; d < D; d++)
                ddelta_user[u][d] += dl1 * (1 - topicProportion) * delta_conv[vi->item][d];
            for (int w = 0; w < embeddingSize; w++)
                dtopo_user[u][w] += dl1 * conversationEmbeddings[vi->item][w];
        }
    }

#pragma omp parallel for reduction(+:da)
    for (int u = 0; u < nUsers; u++) {
        for (vector<int>::iterator it = trainNovotesPerUser[u].begin(); it != trainNovotesPerUser[u].end(); it++) {
            int b = *it;
            double p2 = prediction(u, b);
            double dl2 = dsquare(p2 - 0) * confidence[u][b];

            da += dl2;
            dbeta_user[u] += dl2;
            for (int k = 0; k < K; k++)
                dgamma_user[u][k] += dl2 * topicProportion * gamma_conv[b][k];
            for (int d = 0; d < D; d++)
                ddelta_user[u][d] += dl2 * (1 - topicProportion) * delta_conv[b][d];
            for (int w = 0; w < embeddingSize; w++)
                dtopo_user[u][w] += dl2 * conversationEmbeddings[b][w];

        }
    }
    (*dalpha) = da;

#pragma omp parallel for
    for (int b = 0; b < nConvs; b++) {
        for (vector<vote *>::iterator it = trainVotesPerConv[b].begin(); it != trainVotesPerConv[b].end(); it++) {
            vote *vi = *it;
            double p1 = prediction(vi->user, b);
            double dl1 = dsquare(p1 - RRATING) * confidence[vi->user][b];

            dbeta_conv[b] += dl1;
            for (int k = 0; k < K; k++)
                dgamma_conv[b][k] += dl1 * topicProportion * gamma_user[vi->user][k];
            for (int d = 0; d < D; d++)
                ddelta_conv[b][d] += dl1 * (1 - topicProportion) * delta_user[vi->user][d];
        }
    }

#pragma omp parallel for
    for (int b = 0; b < nConvs; b++) {
        for (vector<int>::iterator it = trainNovotesPerConv[b].begin(); it != trainNovotesPerConv[b].end(); it++) {
            int u = *it;
            double p2 = prediction(u, b);
            double dl2 = dsquare(p2 - 0) * confidence[u][b];

            dbeta_conv[b] += dl2;
            for (int k = 0; k < K; k++)
                dgamma_conv[b][k] += dl2 * topicProportion * gamma_user[u][k];
            for (int d = 0; d < D; d++)
                ddelta_conv[b][d] += dl2 * (1 - topicProportion) * delta_user[u][d];
        }
    }

    double dk = 0;
#pragma omp parallel for reduction(+:dk)
    for (int b = 0; b < nConvs; b++) {
        double tZ;
        topicZ(b, tZ);

        for (int k = 0; k < K; k++) {
            double q = -lambda * (convTopicCounts[b][k] - convTopicSums[b] * exp(*kappa_topic * gamma_conv[b][k]) / tZ);
            dgamma_conv[b][k] += *kappa_topic * q;
            dk += gamma_conv[b][k] * q;
        }
    }
    (*dkappa_topic) = dk;

    double dd = 0;
#pragma omp parallel for reduction(+:dd)
    for (int u = 0; u < nUsers; u++) {
        double dZ;
        discourseZ(u, dZ);

        for (int d = 0; d < D; d++) {
            double p = -lambda *
                       (userDiscourseCounts[u][d] - userDiscourseSums[u] * exp(*kappa_disc * delta_user[u][d]) / dZ);
            ddelta_user[u][d] += *kappa_disc * p;
            dd += delta_user[u][d] * p;
        }
    }
    (*dkappa_disc) = dd;

    double *wtZ = new double[K];
    wordtopicZ(wtZ);
#pragma omp parallel for
    for (int w = 0; w < nWords; w++)
        for (int k = 0; k < K; k++) {
            int twC = wordTopicCounts[w][k];
            double ex1 = exp(bg_topicWords[w] + topicWords[w][k]);
            dtopicWords[w][k] += -lambda * (twC - wordTopicSums[k] * ex1 / wtZ[k]);
        }
    delete[] wtZ;

    double *wdZ = new double[D];
    worddiscourseZ(wdZ);
#pragma omp parallel for
    for (int w = 0; w < nWords; w++)
        for (int d = 0; d < D; d++) {
            int dwC = wordDiscourseCounts[w][d];
            double ex2 = exp(bg_discourseWords[w] + discourseWords[w][d]);
            ddiscourseWords[w][d] += -lambda * (dwC - wordDiscourseSums[d] * ex2 / wdZ[d]);
        }
    delete[] wdZ;

    double wbZ;
    wordbackgroundZ(wbZ);
#pragma omp parallel for
    for (int w = 0; w < nWords; w++) {
        int bwC = wordBackgroundCounts[w];
        double ex3 = exp(backgroundWords[w]);
        dbackgroundWords[w] += -lambda * (bwC - backgroundCounts * ex3 / wbZ);
    }

    double *wyZ = new double[D];
    wordtypeZ(wyZ);
#pragma omp parallel for
    for (int k = 0; k < 3; k++)
        for (int d = 0; d < D; d++) {
            int ywC = discourseWordtypeCounts[d][k];
            double ex4 = exp(typeSwitcher[d][k]);
            dtypeSwitcher[d][k] += -lambda * (ywC - discourseWordtypeSums[d] * ex4 / wyZ[d]);
        }
    delete[] wyZ;

    // Add the derivative of the regularizer
    if (latentReg > 0) {
        for (int u = 0; u < nUsers; u++) {
            for (int k = 0; k < K; k++)
                dgamma_user[u][k] += latentReg * dsquare(gamma_user[u][k]);
            for (int d = 0; d < D; d++)
                ddelta_user[u][d] += latentReg * dsquare(delta_user[u][d]);
        }
        for (int b = 0; b < nConvs; b++) {
            for (int k = 0; k < K; k++)
                dgamma_conv[b][k] += latentReg * dsquare(gamma_conv[b][k]);
            for (int d = 0; d < D; d++)
                ddelta_conv[b][d] += latentReg * dsquare(delta_conv[b][d]);
        }
    }

    clearG(&(dalpha), &(dkappa_disc), &(dkappa_topic), &(dbeta_user), &(dbeta_conv), &(dgamma_user), &(dgamma_conv),
           &(ddelta_user), &(ddelta_conv), &(dtopo_user), &(dtopicWords), &(ddiscourseWords), &(dbackgroundWords), &(dtypeSwitcher));
}

/// Compute the energy according to the least-squares criterion, cost function is MSE
double topicCorpus::lsq() {
    // double lsqStart = clock_();
    double res = 0;
#pragma omp parallel for reduction(+:res)
    for (int x = 0; x < (int) trainVotes.size(); x++) {
        vote *vi = trainVotes[x];
        res += confidence[vi->user][vi->item] * square(prediction(vi->user, vi->item) - RRATING);
    }
    for (int u = 0; u < nUsers; u++) {
        for (vector<int>::iterator it = trainNovotesPerUser[u].begin(); it != trainNovotesPerUser[u].end(); it++) {
            int b = (*it);
            res += confidence[u][b] * square(prediction(u, b) - 0);
        }
    }

    for (int u = 0; u < nUsers; u++) {
        double dZ;
        discourseZ(u, dZ);
        double ldZ = log(dZ);
        for (int d = 0; d < D; d++)
            res += -lambda * userDiscourseCounts[u][d] * (*kappa_disc * delta_user[u][d] - ldZ);
    }

    for (int b = 0; b < nConvs; b++) {
        double tZ;
        topicZ(b, tZ);
        double ltZ = log(tZ);
        for (int k = 0; k < K; k++)
            res += -lambda * convTopicCounts[b][k] * (*kappa_topic * gamma_conv[b][k] - ltZ);
    }

    double *wtZ = new double[K];
    wordtopicZ(wtZ);
    for (int k = 0; k < K; k++) {
        double lwtZ = log(wtZ[k]);
        for (int w = 0; w < nWords; w++)
            res += -lambda * wordTopicCounts[w][k] * (bg_topicWords[w] + topicWords[w][k] - lwtZ);
    }
    delete[] wtZ;

    double *wdZ = new double[D];
    worddiscourseZ(wdZ);
    for (int d = 0; d < D; d++) {
        double lwdZ = log(wdZ[d]);
        for (int w = 0; w < nWords; w++) {
            res += -lambda * wordDiscourseCounts[w][d] * (bg_discourseWords[w] + discourseWords[w][d] - lwdZ);
        }
    }
    delete[] wdZ;

    double wbZ;
    wordbackgroundZ(wbZ);
    double lwbZ = log(wbZ);

    for (int w = 0; w < nWords; w++)
        res += -lambda * wordBackgroundCounts[w] * (backgroundWords[w] - lwbZ);
    double *wyZ = new double[D];
    wordtypeZ(wyZ);
    for (int d = 0; d < D; d++) {
        double lwyZ = log(wyZ[d]);
        for (int t = 0; t < 3; t++)
            res += -lambda * discourseWordtypeCounts[d][t] * (typeSwitcher[d][t] - lwyZ);
    }
    delete[] wyZ;

    // Add the regularizer to the energy
    if (latentReg > 0) {
        for (int u = 0; u < nUsers; u++) {
            for (int k = 0; k < K; k++)
                res += latentReg * square(gamma_user[u][k]);
            for (int d = 0; d < D; d++)
                res += latentReg * square(delta_user[u][d]);
        }
        for (int b = 0; b < nConvs; b++) {
            for (int k = 0; k < K; k++)
                res += latentReg * square(gamma_conv[b][k]);
            for (int d = 0; d < D; d++)
                res += latentReg * square(delta_conv[b][d]);
        }
    }

    // double lsqEnd = clock_();
    return res;
}

/// Compare function for sorting ratings
bool comparison(rating const &a, rating const &b) {
    return a.predict > b.predict;
}

bool comparison2(rating const &a, rating const &b) {
    return a.real > b.real;
}

/// Calculate n_precision for a ranked list
double calPrecision(vector <rating> rankedRatings, int n) {
    double ones = 0;
    int i;
    for (i = 0; i < (int) rankedRatings.size(); i++) {
        if (rankedRatings[i].real == RRATING)
            ones++;
        if (ones == n)
            break;
    }
    return ones / (i + 1);
}

///
double averagePrecision(vector <rating> rankedRatings) {
    double totalNum = 0;
    double ap = 0;
    for (vector<rating>::iterator it = rankedRatings.begin(); it != rankedRatings.end(); it++) {
        totalNum += (*it).real;
    }
    if (totalNum == 0)
        return -1;
    else {
        for (int n = 1; n <= totalNum; n++)
            ap += calPrecision(rankedRatings, n);
        return ap / totalNum;
    }
}

/// Compute evaluation part for conversation data ranking, MAP
void topicCorpus::meanAveragePrecision(double &valid, double &test) {
    map<int, vector<rating> > validRatings;
    map<int, vector<rating> > testRatings;
    double ap;
    int totalNum = 0;
    valid = 0;
    test = 0;

    // Compute MAP of validation set
    for (vector<vote *>::iterator it = validVotes.begin(); it != validVotes.end(); it++) {
        double pred = prediction((*it)->user, (*it)->item);
        validRatings[(*it)->user].push_back(rating(pred, RRATING));
    }
    for (int u = 0; u < nUsers; u++) {
        for (vector<int>::iterator it = validNovotesPerUser[u].begin(); it != validNovotesPerUser[u].end(); it++) {
            int b = (*it);
            validRatings[u].push_back(rating(prediction(u, b), 0));
        }
    }
    printf("here\n");
    for (map < int, vector < rating > > ::iterator it = validRatings.begin(); it != validRatings.end();
    it++)
    {
        sort(validRatings[it->first].begin(), validRatings[it->first].end(), comparison);
        ap = averagePrecision(validRatings[it->first]);
        if (ap != -1) {
            totalNum++;
            valid += ap;
        }
    }
    valid /= totalNum;
    // Compute MAP of test set
    totalNum = 0;
    for (vector<vote *>::iterator it = testVotes.begin(); it != testVotes.end(); it++) {
        testRatings[(*it)->user].push_back(rating(prediction((*it)->user, (*it)->item), RRATING));
    }
    for (int u = 0; u < nUsers; u++) {
        for (vector<int>::iterator it = testNovotesPerUser[u].begin(); it != testNovotesPerUser[u].end(); it++) {
            int b = (*it);
            testRatings[u].push_back(rating(prediction(u, b), 0));
        }
    }
    for (map < int, vector < rating > > ::iterator it = testRatings.begin(); it != testRatings.end();
    it++)
    {
        sort(testRatings[it->first].begin(), testRatings[it->first].end(), comparison);
        ap = averagePrecision(testRatings[it->first]);
        if (ap != -1) {
            totalNum++;
            test += ap;
        }
    }
    test /= totalNum;
}

double calPrecisionAtN(vector <rating> rankedRatings, int n) {
    double ones = 0;
    int i;
    for (i = 0; i < n; i++) {
        if (rankedRatings[i].real == RRATING)
            ones++;
    }
    return ones / n;
}

/// Compute precision@N
double topicCorpus::precisionAtN(map<int, vector<rating> > testRatings, int n) {
    double precision = 0;
    int totalNum = 0;
    for (map < int, vector < rating > > ::iterator it = testRatings.begin(); it != testRatings.end();
    it++)
    {
        double totalValue = 0;
        for (vector<rating>::iterator itt = it->second.begin(); itt != it->second.end(); itt++)
            totalValue += (*itt).real;
        if (totalValue > 0 && (int) it->second.size() >= n) {
            sort(testRatings[it->first].begin(), testRatings[it->first].end(), comparison);
            precision += calPrecisionAtN(testRatings[it->first], n);
            totalNum++;
        }
    }
    precision /= totalNum;
    return precision;
}

double calDCG(vector <rating> rankedRatings, int k) {
    double res = 0.0;
    for (int i = 0; i < k; i++) {
        double numerator = pow(2, rankedRatings[i].real) - 1.0;
        double denominator = log2(2 + i);
        res += numerator / denominator;
    }
    return res;
}

double calNDCG(vector <rating> ratings, int k) {
    sort(ratings.begin(), ratings.end(), comparison);
    double dcg = calDCG(ratings, k);
    sort(ratings.begin(), ratings.end(), comparison2);
    double idcg = calDCG(ratings, k);
    if (idcg == 0)
        return -1;
    return dcg / idcg;
}

/// Compute nDCG@k
double topicCorpus::nDCG(map<int, vector<rating> > testRatings, int k) {
    double ndcg = 0;
    int totalNum = 0;
    for (map < int, vector < rating > > ::iterator it = testRatings.begin(); it != testRatings.end();
    it++)
    {
        if ((int) it->second.size() >= k) {
            double temp = calNDCG(testRatings[it->first], k);
            if (temp != -1) {
                ndcg += temp;
                totalNum++;
            }
        }
    }
    ndcg /= totalNum;
    return ndcg;
}

/// Compute evaluation part for conversation data ranking, Precision@N and nDCG
void topicCorpus::otherEvaluation(double *pnResult, double *ndcgResult, std::string scorePath) {
    map<int, vector<rating> > testRatings;
    FILE *f = fopen_(scorePath.c_str(), "w");

    for (vector<vote *>::iterator it = testVotes.begin(); it != testVotes.end(); it++) {
        int u = (*it)->user;
        int c = (*it)->item;
        double pre = prediction(u, c);
        testRatings[u].push_back(rating(pre, 1));
        fprintf(f, "%s\t%s\t%f\t%d\n", corp->rUserIds[u].c_str(), corp->rConvIds[c].c_str(), pre, 1);
    }
    for (int u = 0; u < nUsers; u++) {
        for (vector<int>::iterator it = testNovotesPerUser[u].begin(); it != testNovotesPerUser[u].end(); it++) {
            int c = (*it);
            double pre = prediction(u, c);
            testRatings[u].push_back(rating(pre, 0));
            fprintf(f, "%s\t%s\t%f\t%d\n", corp->rUserIds[u].c_str(), corp->rConvIds[c].c_str(), pre, 0);
        }
    }
    fclose(f);

    int Ns[4] = {1, 5, 10, 25};
    int Ks[4] = {5, 10, 25, 50};

    for (int i = 0; i < 4; i++)
        pnResult[i] = precisionAtN(testRatings, Ns[i]);
    for (int i = 0; i < 4; i++)
        ndcgResult[i] = nDCG(testRatings, Ks[i]);

}

/// Print out the top words for each topic
void topicCorpus::topWords() {
    printf("Top words for each topic:\n");
    for (int k = 0; k < K; k++) {
        vector <pair<double, int>> bestWords;
        for (int w = 0; w < nWords; w++)
            bestWords.push_back(pair<double, int>(-topicWords[w][k], w));
        sort(bestWords.begin(), bestWords.end());
        for (int w = 0; w < 5; w++) {
            printf("%s(%f) ", corp->idWord[bestWords[w].second].c_str(), -bestWords[w].first);
        }
        printf("\n");
    }
    printf("Top words for each discourse:\n");
    for (int d = 0; d < D; d++) {
        vector <pair<double, int>> bestWords;
        for (int w = 0; w < nWords; w++)
            bestWords.push_back(pair<double, int>(-discourseWords[w][d], w));
        sort(bestWords.begin(), bestWords.end());
        for (int w = 0; w < 5; w++) {
            printf("%s(%f) ", corp->idWord[bestWords[w].second].c_str(), -bestWords[w].first);
        }
        printf("\n");
    }
}

/// Save a model and result to two files
void topicCorpus::save(std::string modelPath, std::string resultPath, std::string scorePath) {
    FILE *f;
    getG(bestW, &(alpha), &(kappa_disc), &(kappa_topic), &(beta_user), &(beta_conv), &(gamma_user), &(gamma_conv),
         &(delta_user), &(delta_conv), &(topo_user), &(topicWords), &(discourseWords), &(backgroundWords), &(typeSwitcher), false);
    if (lambda > 0) {
        f = fopen_(modelPath.c_str(), "w");
        for (int k = 0; k < K; k++) {
            fprintf(f, "Topic %d\n", k);
            vector <pair<double, int>> bestWords;
            for (int w = 0; w < nWords; w++)
                bestWords.push_back(pair<double, int>(-topicWords[w][k], w));
            sort(bestWords.begin(), bestWords.end());
            for (int w = 0; w < 30; w++)
                fprintf(f, "%s %f\n", corp->idWord[bestWords[w].second].c_str(), -bestWords[w].first);
            fprintf(f, "\n");
        }

        for (int d = 0; d < D; d++) {
            fprintf(f, "discourse %d\n", d);
            vector <pair<double, int>> bestWords;
            for (int w = 0; w < nWords; w++)
                bestWords.push_back(pair<double, int>(-discourseWords[w][d], w));
            sort(bestWords.begin(), bestWords.end());
            for (int w = 0; w < 30; w++)
                fprintf(f, "%s %f\n", corp->idWord[bestWords[w].second].c_str(), -bestWords[w].first);
            if (d < D - 1)
                fprintf(f, "\n");
        }
        fclose(f);
    }

    double pnResult[4];
    double ndcgResult[4];
    otherEvaluation(pnResult, ndcgResult, scorePath);

    f = fopen_(resultPath.c_str(), "w");
    fprintf(f,
            "Topic num: %d, Discourse num: %d, topicProportion: %.3f, latentReg: %.3f, lambda: %.3f, c1: %d, c2: %d\n",
            K, D, topicProportion, latentReg, lambda, c1, c2);
    fprintf(f, "Valid MAPscore: %f, Test MAPscore: %f\n", bestValidScore, bestValidTestScore);
    fprintf(f, "P@1: %f, P@5: %f, P@10: %f, P@25: %f\n", pnResult[0], pnResult[1], pnResult[2], pnResult[3]);
    fprintf(f, "nDCG@5: %f, nDCG@10: %f, nDCG@25: %f, nDCG@50: %f\n\n", ndcgResult[0], ndcgResult[1], ndcgResult[2],
            ndcgResult[3]);
    fprintf(f, "alpha: %f, kappa_disc: %f, kappa_topic: %f\n", *alpha, *kappa_disc, *kappa_topic);
    fprintf(f, "Users Parameters: \n");
    for (int u = 0; u < nUsers; u++) {
        fprintf(f, "%s:\t%f  ", corp->rUserIds[u].c_str(), beta_user[u]);
        for (int k = 0; k < K; k++)
            fprintf(f, " %f", gamma_user[u][k]);
        for (int d = 0; d < D; d++)
            fprintf(f, " %f", delta_user[u][d]);
        fprintf(f, "\n");
    }
    fprintf(f, "\nConvs Parameters: \n");
    for (int b = 0; b < nConvs; b++) {
        fprintf(f, "%s:\t%f  ", corp->rConvIds[b].c_str(), beta_conv[b]);
        for (int k = 0; k < K; k++)
            fprintf(f, " %f", gamma_conv[b][k]);
        for (int d = 0; d < D; d++)
            fprintf(f, " %f", delta_conv[b][d]);
        fprintf(f, "\n");
    }
    fprintf(f, "\nTypeSwitcher Parameters: \n");
    for (int d = 0; d < D; d++) {
        for (int t = 0; t < 3; t++)
            fprintf(f, "%f ", typeSwitcher[d][t]);
        fprintf(f, "\n");
    }
    fclose(f);

}

/// Train a model for "emIterations" with "gradIterations" of gradient descent at each step
void topicCorpus::train(int emIterations, int gradIterations) {
    bestValidScore = -1;
    for (int emi = 0; emi < emIterations; emi++) {
        printf("\nIteration time: %d\n", emi + 1);
        lbfgsfloatval_t fx = 0;
        lbfgsfloatval_t *x = lbfgs_malloc(NW);
        for (int i = 0; i < NW; i++)
            x[i] = W[i];

        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.max_iterations = gradIterations;
        param.epsilon = 1e-2;
        param.delta = 1e-2;
        lbfgs(NW, x, &fx, evaluate, progress, (void *) this, &param);
        printf("\nenergy after gradient step = %f\n", fx);
        lbfgs_free(x);

        if (lambda > 0) {
            msgSampling();
            normalizeWordWeights();
            topWords();
        }

        double valid, test;
        meanAveragePrecision(valid, test);
        printf("\nMAP (valid/test) = %f/%f\n", valid, test);

        if (valid > bestValidScore) {
            bestValidScore = valid;
            bestValidTestScore = test;
            for (int i = 0; i < NW; i++)
                bestW[i] = W[i];
        }
    }
}

int main(int argc, char **argv) {
    srand((unsigned) time(0));

    if (argc < 2) {
        printf("An input file is required\n");
        exit(0);
    }

    double latentReg = 0;
    double lambda = 0.1;
    double topicProportion = 0.5;
    int K = 10;
    int D = 10;
    int c1 = 200;
    int c2 = 0;
    double percentage = 0.75;

    std::string modelPath;
    std::string resultPath;
    std::string scorePath;

    if (argc >= 7) {
        lambda = atof(argv[2]);
        topicProportion = atof(argv[3]);
        K = atoi(argv[4]);
        D = atoi(argv[5]);
        percentage = atof(argv[6]);
    }

    if (argc >= 9) {
        c1 = atoi(argv[7]);
        c2 = atoi(argv[8]);
    }

    if (argc == 12) {
        modelPath = argv[9];
        resultPath = argv[10];
        scorePath = argv[11];
    } else {
        stringstream strStream;
        strStream << "_" << lambda << "_" << topicProportion << "_" << K << "_" << D << "_" << percentage << "_" << c1
                  << "_" << c2 << "_";
        string s = strStream.str();
        modelPath = argv[1];
        resultPath = argv[1];
        scorePath = argv[1];
        modelPath.append(s);
        resultPath.append(s);
        scorePath.append(s);
        modelPath.append("model.out");
        resultPath.append("result.out");
        scorePath.append("score.out");
    }

    printf("corpus = %s\n", argv[1]);
    printf("latentReg = %f\n", latentReg);
    printf("lambda = %f\n", lambda);
    printf("K = %d\n", K);
    printf("D = %d\n", D);
    printf("percentage = %f\n", percentage);
    printf("topicProportion = %.3f\n", topicProportion);
    printf("c1 = %d\n", c1);

    corpus corp(argv[1], percentage);
    topicCorpus ec(&corp, K, D, topicProportion, // K and D and topic proportion
                   latentReg, // latent topic regularizer
                   lambda,  // lambda
                   c1, c2);
    ec.train(20, 80);
    ec.save(modelPath, resultPath, scorePath);

    return 0;
}
