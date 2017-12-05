import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

def readInAndConvert():
    '''
    Reads in .dat file through pandas, converts to numpy
    :return: Numpy array nusers X nmovies, cells are ratings
    '''
    start_time = time.time()

    print "Loading Dataset..."

    my_data = pd.read_table('ml-1m/ratings.dat', sep="::", header = None )
    print my_data.head(10)

    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    print "Converting to Numpy"
    my_data = my_data[[0, 1, 2]].as_matrix()
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    print "Converting Dataset to Desired Format..."
    mat = convert_matrix(my_data) # convert to required matrix, q1


    print("--- %s seconds to Convet & make Sparse---" % (time.time() - start_time))
    return mat


def convert_matrix(my_data):
    '''

    :param my_data: pandas df
    :return: numpy matrix
    '''
    nmovies = np.amax(my_data, axis=0)[1]
    nusers = np.amax(my_data, axis=0)[0]
    print "Users", nusers, "Movies", nmovies
    df = np.zeros((int(nmovies), int(nusers)),dtype=np.uint8)
    for i in range(int(np.shape(my_data)[0])):
        if my_data[i, 2] > 0:
            df[int(my_data[i, 1] - 1), int(my_data[i, 0] - 1)] = my_data[i, 2]
    return df.T


def findMaxMovie(mat,subset,indicesToCheck,maxA):
    '''
    Helper function for greedy search, finds argmax of j for F(A)
    :param mat: Numpy matrix nusers X nmovies
    :param subset: set of movies already chosen
    :param indicesToCheck: valid indices to check, ie movies not in th set
    :param maxA: returns column corresponding to max rating in subset A for each user
    :return:
    '''

    bestScore = 0
    maxIndex = 0


    for i in indicesToCheck:
        # print np.shape(maxA)
        # print np.shape(mat[:,i])
        canidateMaxA = np.max(np.concatenate((maxA.reshape(-1,1),mat[:,i].reshape(-1,1)),axis = 1), axis = 1)
        score = np.mean(canidateMaxA, axis=0)


        #print score
        if score >= bestScore:
            bestScore = score
            maxIndex = i
            bestCanidateMaxA = canidateMaxA


    return maxIndex, bestCanidateMaxA




def lazySearch(mat,kValues):

    start_time = time.time()
    times = []
    k = max(kValues)
    subset = []
    scores = []
    # get first score
    marginals = np.mean(mat,axis=0)

    # choose first movie index
    sortedMarginals = np.sort(marginals)[::-1]
    sortedIndexes = np.argsort(marginals)[::-1]

    # get subset cols
    maxA = mat[:, sortedIndexes[0]]
    subset.append(sortedIndexes[0])

    #Remove used index from availables
    sortedIndexes = sortedIndexes[1:]
    sortedMarginals = sortedMarginals[1:]

    i = 0

    while len(subset) < k:
        # for i in sortedIndexes:
        # print sortedIndexes[i]
        canidateMaxA = np.max(np.concatenate((maxA.reshape(-1, 1), mat[:, sortedIndexes[i]].reshape(-1, 1)), axis=1), axis=1)
        score = np.mean(canidateMaxA, axis=0)

        marginalScore = (score - np.mean(maxA,axis=0))

        if marginalScore >= sortedMarginals[1]:
            # Update subset and MaxA
            subset.append(sortedIndexes[i])
            maxA = canidateMaxA
            # print subset
            if len(subset) in kValues:
                scores.append(np.mean(maxA))
                times.append(time.time() - start_time)

            # Remove from available
            indsToRemove = np.where(sortedIndexes != sortedIndexes[i])
            sortedIndexes = sortedIndexes[indsToRemove]
            sortedMarginals = sortedMarginals[indsToRemove]

        else:
            sortedMarginals[0] = marginalScore
            argSortMarginals = np.argsort(sortedMarginals)[::-1]

            sortedIndexes = sortedIndexes[argSortMarginals]
            sortedMarginals = sortedMarginals[argSortMarginals]

    return subset, scores, times


def greedySearch(mat,kValues):
    start_time = time.time()
    subset = []
    times = []
    k = max(kValues)
    indicesToCheck = np.arange(0, np.shape(mat)[1])
    firstMovie = getFirstMovie(mat)
    maxA = mat[:,firstMovie]
    subset.append(firstMovie)
    indicesToCheck = indicesToCheck[np.where(indicesToCheck != firstMovie)]

    scores = []
    for i in range(k-1):

        # Find best movie
        maxMovie, maxA = findMaxMovie(mat, subset,indicesToCheck, maxA)

        # Remove Movie from Indices and Add to Set

        indicesToCheck = indicesToCheck[np.where(indicesToCheck != maxMovie)]
        subset.append(maxMovie)
        if len(subset) in kValues:
            scores.append(np.mean(maxA))
            times.append(time.time() - start_time)

    return subset, scores, times

def getFirstMovie(mat):
    return np.argmax(np.sum(mat,axis=0))


def plotValues(greedy,lazy, kValues):
    plt.plot(kValues, greedy,'r',marker = 'o',label="Greedy")
    plt.plot(kValues, lazy, 'b--', linestyle = '--', marker = 'x', label="lazy")
    plt.legend(loc='best')
    plt.xlabel('K-Value')
    plt.ylabel('Objective Function Value ')
    plt.savefig('Scores.png')
    plt.close()


def plotTimes(greedy, lazy, kValues):
    plt.plot(kValues, greedy, 'r',marker = 'o', label="Greedy")
    plt.plot(kValues, lazy, 'b--', marker = 'o', label="lazy")
    plt.legend(loc='best')
    plt.xlabel('K-Value')
    plt.ylabel('Runtime in Seconds')
    plt.savefig('Times.png')
    plt.close()



if __name__ == "__main__":
    total_time = time.time()

    # mat = readInAndConvert()
    # np.save('mat.npy',mat)

    mat = np.load('mat.npy')
    kValues = [10,20,30,40,50,100,200,500]


    setGreedy, scoreGreedy, timesGreedy  = greedySearch(mat,kValues)
    print "Set: ", (setGreedy)
    print "Score: ", scoreGreedy
    print "Runtimes: ", timesGreedy

    setLazy, scoreLazy, timesLazy  = lazySearch(mat,kValues)

    print "Set: ", (setLazy)
    print "Score: ", scoreLazy
    print "Runtimes: ", timesLazy



    plotValues(scoreGreedy, scoreLazy, kValues)
    plotTimes(timesGreedy, timesLazy, kValues)


    print("--- %s seconds ---" % (time.time() - total_time))
