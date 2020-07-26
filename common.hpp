#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "math.h"
#include "string.h"
#include <string>
#include <iostream>
#include "omp.h"
#include "map"
#include "set"
#include "common.hpp"
#include "algorithm"
#include "lbfgs.h"
#include "sstream"
#include "gzstream.h"

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
  FILE* f = fopen(p, m);
  if (!f)
  {
    printf("Failed to open %s\n", p);
    exit(1);
  }
  return f;
}

/// Data associated with a rating
struct vote
{
  int user; // ID of the user
  int item; // ID of the item

  std::vector<int> words; // IDs of the words in the review
};

typedef struct vote vote;

/// To sort words by frequency in a corpus
bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2)
{
  return p1.second > p2.second;
}

/// To sort votes by product ID
bool voteCompare(vote* v1, vote* v2)
{
  return v1->item > v2->item;
}

/// split the string s using string c
void splitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

/// Sign (-1, 0, or 1)
template<typename T> int sgn(T val)
{
  return (val > T(0)) - (val < T(0));
}

class corpus
{
public:
  corpus(std::string voteFile, double percentage)
  {
    srand((unsigned)time(0));
    std::map<std::string, std::vector<std::string> > uCounts;
    std::map<std::string, int> cCounts;
    std::string uName;
    std::string cName;
    int nRead = 0;

    igzstream in;
    in.open(voteFile.c_str());
    std::string line;
    std::string sWord;
    std::vector<std::string> msgs;
    std::vector<std::string> words;
    std::string msgID;
    std::map<std::string, std::vector<std::string> > msgIDs;
    std::vector<std::string> msgIDs_test;
    
    // Read the input file. The first time the file is read it is only to compute word counts, in order to select the top "maxWords" words to include in the dictionary
    while (std::getline(in, line))
    {
      splitString(line, msgs, "\t");
      cName = msgs[0];
      uName = msgs[5];
      msgID = msgs[1];
      msgIDs[cName].push_back(msgID);
      splitString(msgs[4], words, " ");
      for (int w = 0; w < (int) words.size(); w++)
      {
        sWord = words[w];
        if (wordCount.find(sWord) == wordCount.end())
          wordCount[sWord] = 0;
        wordCount[sWord]++;
      }
      if (cCounts.find(cName) == cCounts.end())
        cCounts[cName] = 0;
      uCounts[uName].push_back(cName);
      cCounts[cName]++;

      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush(stdout);
      }
      msgs.clear();
      words.clear();

    }
    in.close();

    printf("\nnUsers = %d, nItems = %d, nRatings = %d\n", (int) uCounts.size(), (int) cCounts.size(), nRead);

    V = new std::vector<vote*>();
    vote* v = new vote();
    std::map<std::string, int> userIds;
    std::map<std::string, int> convIds;

    nUsers = 0;
    nConvs = 0;
    // Comment this block to include all users, otherwise only users/items with userMin/convMin ratings will be considered
    //    nUsers = 1;
    //    nConvs = 1;
    //    userIds["NOT_ENOUGH_VOTES"] = 0;
    //    convIds["NOT_ENOUGH_VOTES"] = 0;
    //    rUserIds[0] = "NOT_ENOUGH_VOTES";
    //    rConvIds[0] = "NOT_ENOUGH_VOTES";
    //    vote* v_ = new vote();
    //    v_->user = 0;
    //    v_->item = 0;
    //    v_->value = 0;
    //    v_->voteTime = 0;
    //    V->push_back(v_);

    int userMin = 0;
    int convMin = 0;

    int maxWords = 5000; // Dictionary size
    std::vector < std::pair<std::string, int> > whichWords;
    for (std::map<std::string, int>::iterator it = wordCount.begin(); it != wordCount.end(); it++)
      whichWords.push_back(*it);
    sort(whichWords.begin(), whichWords.end(), wordCountCompare);
    if ((int) whichWords.size() < maxWords)
      maxWords = (int) whichWords.size();
    nWords = maxWords;
    for (int w = 0; w < maxWords; w++)
    {
      wordId[whichWords[w].first] = w;
      idWord[w] = whichWords[w].first;
    }

    // Newest 20% of every conversation as valid and test set
    for (std::map<std::string, std::vector<std::string> >::iterator it = msgIDs.begin(); it != msgIDs.end(); it++)
    {
      int convSize = (int) (it->second).size();
      if (convSize <= 1)
        continue;
      else
      {
        int choose = (int)(convSize * percentage);
        if (choose == 0)
          choose ++;
        else if (choose == convSize)
          choose --;
        for (int n = choose; n < convSize; n ++)
          msgIDs_test.push_back(it->second[n]);
      }

    }
    trainVotesPerUser = new std::vector<vote*>[(int) uCounts.size()];
    trainVotesPerConv = new std::vector<vote*>[(int) cCounts.size()];

    // Re-read the entire file, this time building structures from those words in the dictionary
    igzstream in2;
    in2.open(voteFile.c_str());
    nRead = 0;
    while (std::getline(in2, line))
    {
      splitString(line, msgs, "\t");
      cName = msgs[0];
      uName = msgs[5];
      msgID = msgs[1];
      splitString(msgs[4], words, " ");
      for (int w = 0; w < (int) words.size(); w++)
      {
        sWord = words[w];
        if (wordId.find(sWord) != wordId.end())
          v->words.push_back(wordId[sWord]);
      }
      if ((int) uCounts[uName].size() >= userMin)
      {
        if (userIds.find(uName) == userIds.end())
        {
          rUserIds[nUsers] = uName;
          userIds[uName] = nUsers++;
        }
        v->user = userIds[uName];
      }
      else
        v->user = 0;

      if (cCounts[cName] >= convMin)
      {
        if (convIds.find(cName) == convIds.end())
        {
          rConvIds[nConvs] = cName;
          convIds[cName] = nConvs++;
        }
        v->item = convIds[cName];
      }
      else
        v->item = 0;

      V->push_back(v);

      // Split the votes to training, vaild and test set
      if (std::find(msgIDs_test.begin(), msgIDs_test.end(), msgID) != msgIDs_test.end())
      {
        int r = rand() % 2;
        if (r == 0)
          testVotes.push_back(v);
        else
          validVotes.push_back(v);
      }
      else
      {
        trainVotes.push_back(v);
        trainVotesPerUser[v->user].push_back(v);
        trainVotesPerConv[v->item].push_back(v);
      }

      v = new vote();
      msgs.clear();
      words.clear();

      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush( stdout);
      }
    }

    nRead = 0;
    for (std::map<std::string, int>::iterator it1 = userIds.begin(); it1 != userIds.end(); it1++)
    {
      uName = it1->first;
      for (std::map<std::string, int>::iterator it2 = convIds.begin(); it2 != convIds.end(); it2++)
      {
        cName = it2->first;
        if (std::find(uCounts[uName].begin(), uCounts[uName].end(), cName) != uCounts[uName].end())
        {
          int ur = userIds[uName];
          int cv = convIds[cName];
          userHasreplied[ur].push_back(cv);
        }
      }
      nRead++;
      if (nRead % 100 == 0)
      {
        printf(".");
        fflush( stdout);
      }
    }
  
    delete v;
    in2.close();

    igzstream in3;
    in3.open("stopwords.txt");
    // Read stopwords file and store the stopwords in stopWords.
    while (std::getline(in3, line))
    {
      std::map<std::string,int>::iterator it;
      it = wordId.find(line);
      if (it != wordId.end())
        stopWords.push_back(it->second);
    }
    in3.close();
    printf("\nStopwords: %d\n", (int) stopWords.size());
    printf("\n");

  }

  ~corpus()
  {
    for (std::vector<vote*>::iterator it = V->begin(); it != V->end(); it++)
      delete *it;
    delete V;
    delete[] trainVotesPerConv;
    delete[] trainVotesPerUser;
  }

  std::vector<vote*>* V;
  std::map<int, std::vector<int> > userHasreplied; // Store the conv numbers that each user has replied

  int nUsers; // Number of users
  int nConvs; // Number of items
  int nWords; // Number of words

  std::map<std::string, int> userIds; // Maps a user's string-valued ID to an integer
  std::map<std::string, int> convIds; // Maps an item's string-valued ID to an integer

  std::map<int, std::string> rUserIds; // Inverse of the above map
  std::map<int, std::string> rConvIds;

  std::map<std::string, int> wordCount; // Frequency of each word in the corpus
  std::map<std::string, int> wordId; // Map each word to its integer ID
  std::map<int, std::string> idWord; // Inverse of the above map
  std::vector<int> stopWords; // Store stopwords

  std::vector<vote*> trainVotes;
  std::vector<vote*> validVotes;
  std::vector<vote*> testVotes;

  std::vector<vote*>* trainVotesPerConv; // Vector of votes for each item, only from the training set
  std::vector<vote*>* trainVotesPerUser; // Vector of votes for each user, only from the training set

};
