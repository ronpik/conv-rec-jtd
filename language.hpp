#include "common.hpp"
#include "map"
#define BACK 0
#define DISC 1
#define TOPIC 2
#define RRATING 1
using namespace std;

class rating {
public:
    float predict, real;
    rating(float m, float n): predict(m), real(n) {}
};

class topicCorpus
{
public:
  topicCorpus(corpus* corp, // The corpus
              int K, // The number of latent factors
              int D, // The number of discourse kinds
              double topicProportion, // The proportion for topic factor(0~1)
              double latentReg, // Parameter regularizer used by the "standard" recommender system
              double lambda, // Word regularizer used by HFT
              int c1,
              int c2) : 
    corp(corp), K(K), D(D), topicProportion(topicProportion), latentReg(latentReg), lambda(lambda), c1(c1), c2(c2)
    {
    srand((unsigned)time(0));
    nUsers = corp->nUsers;
    nConvs = corp->nConvs;
    nWords = corp->nWords;
    stopWords = corp->stopWords;
    testVotes = corp->testVotes;
    validVotes = corp->validVotes;
    trainVotes = corp->trainVotes;
    trainVotesPerUser = corp->trainVotesPerUser;
    trainVotesPerConv = corp->trainVotesPerConv;
    conversationEmbeddings = corp -> convEmbeddings;
    embeddingSize = corp -> embeddingSize;

    double testFraction = 0.1;
    testNovotesPerUser = new std::vector<int>[nUsers];
    validNovotesPerUser = new std::vector<int>[nUsers];
    trainNovotesPerUser = new std::vector<int>[nUsers];
    trainNovotesPerConv = new std::vector<int>[nConvs];

    for (int u = 0; u < nUsers; u++)
    {
      int validNum = 0;
      int testNum = 0;
      for (int b = 0; b < nConvs; b++)
      {
        if (std::find(corp->userHasreplied[u].begin(), corp->userHasreplied[u].end(), b) == corp->userHasreplied[u].end())
        {
          double r = rand() * 1.0 / RAND_MAX;
          if (r < testFraction)
          {
            if (testNum < 100)
            {
              testNovotesPerUser[u].push_back(b);
              testNum += 1;
            }
          }
          else if (r < 2*testFraction)
          {
            if (validNum < 100)
            {
              validNovotesPerUser[u].push_back(b);
              validNum += 1;
            }
          }
          else
          {
            trainNovotesPerUser[u].push_back(b);
            trainNovotesPerConv[b].push_back(u);
          }
        }
      }
    }

    confidence = new int*[nUsers];
    for (int i = 0; i < nUsers; i++)
        confidence[i] = new int[nConvs];
    for (int i = 0; i < nUsers; i++)
        for (int j = 0; j < nConvs; j++)
            confidence[i][j] = 1;
    for (int b = 0; b < nConvs; b++)
      for (vector<vote*>::iterator it = trainVotesPerConv[b].begin(); it != trainVotesPerConv[b].end(); it ++)
      {
        //if (it + 1 == trainVotesPerConv[b].end())
          confidence[(*it)->user][b] = c1;
        //else
          //confidence[(*it)->user][b] = c2;
      }
        // total number of parameters
    NW =
            1                           // model prior
            + 2                         // user bias and conversation bias parameters
            + ((K + D + 1) * nConvs)    // conversation relatedness to topics, dicourse modes and background
            + ((K + D + 1) * nUsers)    // users preferences of topics, dicourse modes and background
            + ((K + D + 1) * nWords)    // words relatedness to topics, discourse modes and background
            + (D * 3)                   // discourse mode switcher parameters
            + (corp->embeddingSize * corp->nUsers); // conversation structure preferences

    // Initialize parameters and latent variables
    // Zero all weights
    W = new double [NW];
    bestW = new double [NW];
    for (int i = 0; i < NW; i++)
      W[i] = 0;
    getG(W, &alpha, &kappa_disc, &kappa_topic, &beta_user, &beta_conv, &gamma_user, &gamma_conv,
            &delta_user, &delta_conv, &topo_user, &topicWords, &discourseWords, &backgroundWords, &typeSwitcher,
            true);

    // Set alpha to the average
    int trainsize = (int) trainVotes.size();
    *alpha += (int) trainVotes.size();
    for (int u = 0; u < nUsers; u++)
    {
      trainsize += (int) trainNovotesPerUser[u].size();
    }
    *alpha /= trainsize;
    double valid, test;
    meanAveragePrecision(valid, test);
    printf("MAP(offset term only) (valid/test) = %f/%f\n", valid, test);
  
    // Set beta to user and product offsets
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      vote* v = *vi;
      beta_user[v->user] += RRATING - *alpha;
      beta_conv[v->item] += RRATING - *alpha;
    }
    for (int u = 0; u < nUsers; u++)
    {
      beta_user[u] -= *alpha * trainNovotesPerUser[u].size();
      beta_user[u] /= (trainVotesPerUser[u].size() + trainNovotesPerUser[u].size());
    }
    for (int b = 0; b < nConvs; b++)
    {
      beta_conv[b] -= *alpha * trainNovotesPerConv[b].size();
      beta_conv[b] /= (trainVotesPerConv[b].size() + trainNovotesPerConv[b].size());
    }
    meanAveragePrecision(valid, test);
    printf("MAP(offset and bias) (valid/test) = %f/%f\n", valid, test);

    // Actually the model works better if we initialize none of these terms?
    /*
    if (lambda > 0)
    {
      *alpha = 0;
      for (int u = 0; u < nUsers; u++)
        beta_user[u] = 0;
      for (int b = 0; b < nConvs; b++)
        beta_conv[b] = 0;
    }
    */
    wordTopicCounts = new int*[nWords];
    wordDiscourseCounts = new int*[nWords];
    wordBackgroundCounts = new int[nWords];
    for (int w = 0; w < nWords; w++)
    {
      wordTopicCounts[w] = new int[K];
      wordDiscourseCounts[w] = new int[D];
      wordBackgroundCounts[w] = 0;
      for (int k = 0; k < K; k++)
        wordTopicCounts[w][k] = 0;
      for (int d = 0; d < D; d++)
        wordDiscourseCounts[w][d] = 0;
    }

    wordTopicSums = new long long[K];
    for (int k = 0; k < K; k++)
    {
      wordTopicSums[k] = 0;
    }
    wordDiscourseSums = new long long[D];
    for (int d = 0; d < D; d++)
    {
      wordDiscourseSums[d] = 0;
    }
    backgroundCounts = 0;

    convTopicCounts = new int*[nConvs];
    convTopicSums = new long long[nConvs];
    for (int b = 0; b < nConvs; b ++) {
        convTopicSums[b] = 0;
        convTopicCounts[b] = new int[K];
        for (int k = 0; k < K; k++)
            convTopicCounts[b][k] = 0;
    }

    userDiscourseCounts = new int*[nUsers];
    userDiscourseSums = new long long[nUsers];
    for (int u = 0; u < nUsers; u ++)
    {
      userDiscourseSums[u] = 0;
      userDiscourseCounts[u] = new int[D];
      for (int d = 0; d < D; d ++)
        userDiscourseCounts[u][d] = 0;
    }

    discourseWordtypeSums = new long long[D];
    for (int d = 0; d < D; d ++)
      discourseWordtypeSums[d] = 0; 
    discourseWordtypeCounts = new long*[D]; 
    for (int d = 0; d < D; d ++)
    {
      discourseWordtypeCounts[d] = new long[3];
      for (int t = 0; t < 3; t ++)
        discourseWordtypeCounts[d][t] = 0;
    }


    // Generate random topic and discourse assignments
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      vote* v = *vi; 
      int k = rand() % K;
      msgTopics[v] = k;
      int d = rand() % D;
      msgDiscourses[v] = d;
      msgWordtypes[v] = new int[(int) v->words.size()];
      convTopicCounts[v->item][k] += 1;
      userDiscourseCounts[v->user][d] += 1;
      convTopicSums[v->item] += 1;
      userDiscourseSums[v->user] += 1;

      for (int wp = 0; wp < (int) v->words.size(); wp++)
      {
        int wi = v->words[wp];
        int t;
        if (std::find(stopWords.begin(), stopWords.end(), wi) != stopWords.end())
          t = rand() % 2;
        else
          t = rand() % 3;
        msgWordtypes[v][wp] = t;
        discourseWordtypeSums[d] += 1;
        discourseWordtypeCounts[d][t] += 1;

        if (t == BACK)
        {
          backgroundCounts += 1;
          wordBackgroundCounts[wi] += 1;
        }
        else if (t == DISC)
        {
          wordDiscourseSums[d] += 1;
          wordDiscourseCounts[wi][d] += 1;
        }
        else if (t == TOPIC)
        {
          wordTopicSums[k] += 1;
          wordTopicCounts[wi][k] += 1;
        }
        else
          printf("Topic or Discourse assigning error!\n");
      }
    }


    // Initialize the topicWords, discourseWords, backgroundWords as word frequency
    long long totalWords = 0;
    bg_topicWords = new double[nWords];
    bg_discourseWords = new double[nWords];
    for (int w = 0; w < nWords; w ++)
      backgroundWords[w] = 0;
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      for (std::vector<int>::iterator it = (*vi)->words.begin(); it != (*vi)->words.end(); it++)
      {
        totalWords++;
        backgroundWords[*it]++;
      }
    }
    for (int w = 0; w < nWords; w++)
    {
      backgroundWords[w] /= totalWords;
      bg_discourseWords[w] = backgroundWords[w];
      bg_topicWords[w] = backgroundWords[w];
    }

    if (lambda == 0)
    {
      for (int u = 0; u < nUsers; u++)
      {
        for (int k = 0; k < K; k++)
          gamma_user[u][k] = rand() * 1.0 / RAND_MAX;
        for (int d = 0; d < D; d++)
          delta_user[u][d] = rand() * 1.0 / RAND_MAX;
        for (int i = 0; i < embeddingSize; i++)
            topo_user[u][i] = rand() * 1.0 / RAND_MAX;
      }
      for (int b = 0; b < nConvs; b++)
      {
        for (int k = 0; k < K; k++)
          gamma_conv[b][k] = rand() * 1.0 / RAND_MAX;
        for (int d = 0; d < D; d++)
          delta_conv[b][d] = rand() * 1.0 / RAND_MAX;
      }
    }
    else
    {
      for (int w = 0; w < nWords; w++)
      {
        for (int k = 0; k < K; k++)
          topicWords[w][k] = 0;
        for (int d = 0; d < D; d++)
          discourseWords[w][d] = 0;
      }
    }
    
    if (lambda > 0)
      msgSampling();

    *kappa_disc = 1.0;
    *kappa_topic = 1.0;
    }

  ~topicCorpus()
  {
    delete[] trainNovotesPerUser;
    delete[] trainNovotesPerConv;
    delete[] testNovotesPerUser;
    delete[] validNovotesPerUser;

    for (int i = 0; i < nUsers; i++)
      delete[] confidence[i];
    delete[] confidence;

    for (int w = 0; w < nWords; w ++)
    {
      delete[] wordTopicCounts[w];
      delete[] wordDiscourseCounts[w];
    }
    delete[] wordTopicCounts;
    delete[] wordDiscourseCounts;
    delete[] wordBackgroundCounts;

    for (int b = 0; b < nConvs; b ++)
      delete[] convTopicCounts[b];
    delete[] convTopicCounts;
    for (int u = 0; u < nUsers; u ++)
      delete[] userDiscourseCounts[u];
    delete[] userDiscourseCounts;
    delete[] wordTopicSums;
    delete[] wordDiscourseSums;
    delete[] convTopicSums;
    delete[] userDiscourseSums;

    for (int d = 0; d < D; d ++)
      delete[] discourseWordtypeCounts[d];
    delete[] discourseWordtypeCounts;
    delete[] discourseWordtypeSums;

    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
      delete[] msgWordtypes[*vi];

    clearG(&alpha, &kappa_disc, &kappa_topic, &beta_user, &beta_conv, &gamma_user, &gamma_conv, &delta_user, &delta_conv, &topo_user, &topicWords, &discourseWords, &backgroundWords, &typeSwitcher);
    delete[] W;
  }

  double prediction(int user, int conv);

  void dl(double* grad);
  void train(int emIterations, int gradIterations);
  double lsq(void);
  void meanAveragePrecision(double& valid, double& test);
  double precisionAtN(map<int, vector<rating> > testRatings, int n);
  double nDCG(map<int, vector<rating> > testRatings, int k);
  void otherEvaluation(double* pnResult, double* ndcgResult, std::string scorePath);
  void save(std::string modelPath, std::string resultPath, std::string scorePath);

  corpus* corp;
  
  // Votes from the training, validation, and test sets
  std::vector<vote*> trainVotes;
  std::vector<vote*> validVotes;
  std::vector<vote*> testVotes;

  double* bestW;
  double bestValidScore;
  double bestValidTestScore;

  std::vector<vote*>* trainVotesPerConv; // Vector of votes for each item, only from the training set
  std::vector<vote*>* trainVotesPerUser; // Vector of votes for each user, only from the training set

  std::vector<int>* testNovotesPerUser; // Store the records that vote 0(Not replied) as test set
  std::vector<int>* validNovotesPerUser; // Store the records that vote 0(Not replied) as valid set
  std::vector<int>* trainNovotesPerUser; // Store the records that vote 0(Not replied) as train set
  std::vector<int>* trainNovotesPerConv; // Store the records that vote 0(Not replied) as train set
  int ** confidence;

  double** conversationEmbeddings; // stores the pre-calculated embedding of each conversation
  int embeddingSize;

  int getG(double* g,
           double** alpha,
           double** kappa_disc,
           double** kappa_topic,
           double** beta_user,
           double** beta_conv,
           double*** gamma_user,
           double*** gamma_conv,
           double*** delta_user,
           double*** delta_conv,
           double*** topo_user,
           double*** topicWords,
           double*** discourseWords,
           double** backgroundWords,
           double*** typeSwitcher,
           bool init);
  void clearG(double** alpha,
              double** kappa_disc,
              double** kappa_topic,
              double** beta_user,
              double** beta_conv,
              double*** gamma_user,
              double*** gamma_conv,
              double*** delta_user,
              double*** delta_conv,
              double*** topo_user,
              double*** topicWords,
              double*** discourseWords,
              double** backgroundWords,
              double*** typeSwitcher);

  void wordtopicZ(double* res);
  void worddiscourseZ(double* res);
  void wordbackgroundZ(double& res);
  void wordtypeZ(double* res);
  void topicZ(int conv, double& res);
  void discourseZ(int user, double& res);
  void msgSampling();
  void topWords();
  void normalizeWordWeights();

  // Model parameters
  double* alpha; // Offset parameter
  double* kappa_topic; // "peakiness" parameter for topic
  double* kappa_disc; // "peakiness" parameter for discourse
  double* beta_user; // User offset parameters
  double* beta_conv; // Item offset parameters
  double** gamma_user; // User latent factors
  double** gamma_conv; // Item latent factors
  double** delta_user; // User latent discourse factors
  double** delta_conv; // Item latent discourse factors
  double** topo_user; // user preferences for conversation topology

  double* W; // Contiguous version of all parameters, i.e., a flat vector containing all parameters in order (useful for lbfgs)

  double** typeSwitcher; // [D][3], weights in certain discourse for three kinds of words, 0 for backgroundword, 1 for discourse word, 2 for topic word
  double** topicWords; // Weights each word in each topic
  double** discourseWords; // Weights each word in each discourse
  double* backgroundWords; // Weights each word in backgroundwords  
  double* bg_topicWords;
  double* bg_discourseWords;
  // Latent variables
  std::map<vote*, int> msgTopics;  // Topic assignment for each message
  std::map<vote*, int> msgDiscourses;  // Discourse assignment for each message
  std::map<vote*, int*> msgWordtypes;  // Wordtype assignment for each word in each message

  // Counters
  int** convTopicCounts; // How many times does each topic occur for each item?
  long long* convTopicSums; // How many times of messages does each topic occur?
  int** userDiscourseCounts; // How many times does each discourse occur for each user?
  long long* userDiscourseSums; // How many times of messages does each discourse occur?
  long long* wordTopicSums; // How many times of words does each topic occur?
  int** wordTopicCounts; // How many times does this topic occur for this word?
  long long* wordDiscourseSums; // How many times of words does each discourse occur?
  int** wordDiscourseCounts; // How many times does this discourse occur for this word?
  long long backgroundCounts; // How many times does background word occur?
  int* wordBackgroundCounts; // How many times does background occur for this word?
  long long* discourseWordtypeSums; // How many times does each discourse occur in all wordtypes?
  long** discourseWordtypeCounts; // How many times does each word type occur for this discourse?

  int NW;
  int K;
  int D;
  double topicProportion;

  double latentReg;
  double lambda;
  int c1;
  int c2;

  std::vector<int> stopWords; // Store stopwords

  int nUsers; // Number of users
  int nConvs; // Number of items
  int nWords; // Number of words
};
