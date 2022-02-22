import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
import pyglet
import math
import imageio
import os


#creating the data frame
file = open('data.txt', 'r')
lines = file.readlines()
sample_size = len(lines)
feature_size = len(lines[0].split())

data = []
images_for_animation = []

for line in lines:
    data.append([float(number) for number in line.split()])

data_frame = pd.DataFrame(data)


#normalizing data (min-max scaling)
for i in range(feature_size):
    minimum = data_frame[i].min()
    maximum = data_frame[i].max()

    data_frame[i] = (data_frame[i] - minimum)/(maximum - minimum)


#centering the data (i.e. making the mean 0), to help computing the covariance matrix
data_centered = data_frame.apply(lambda a: a-a.mean())


#creating 'figures' named folder to save plot images
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'figures')

if not os.path.exists(final_directory):
   os.makedirs(final_directory)



#you can look at the scatter plot and get an idea about the data
def plot_data(samples):
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D Projection')
    X = samples[:, 0]
    Y = samples[:, 1]
    plt.scatter(X, Y, color='g', edgecolors='k')
    plt.show()


#draw K clusters with K different colors
def draw_clusters(samples, posterior, figure):
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D Projection')

    colors = ['g', 'b', 'r']

    for j in range(sample_size):
        probabilities_of_clusters = list(posterior[j])
        cluster = probabilities_of_clusters.index(max(probabilities_of_clusters))
        plt.scatter(samples[j, 0], samples[j, 1], color=colors[cluster])

    plt.savefig('figures/figure' + str(figure) + '.png')
    images_for_animation.append(imageio.imread('figures/figure' + str(figure) + '.png'))



#Principal Component Analysis

#This function returns the first two principal components, then we project our
#data along the new dimensions
def determine_principal_components():

    #1 - compute the covariance matrix, I have used the library function for simplicity
    covariance_matrix = np.cov(data_centered, rowvar=False)

    #2 - determine eigen values and eigen vectors
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    #eigen values are in ascending order in the list in here
    #and according to the documentation, the V[:, i] eigen vector (i.e. the columns resemble the eigen vectors)
    #is associated with the i-th eigen value
    #so, selecting the last two columns as principal components
    pc1 = eigen_vectors[:, -1]
    pc2 = eigen_vectors[:, -2]

    #we got our new dimensions! now, reshaping it for projecting the data later
    new_dimension = np.stack((pc1, pc2), axis=1)      # d x 2

    return new_dimension


new_dimensions = determine_principal_components()
new_data = np.dot(data_frame, new_dimensions)   # n x 2
# plot_data(new_data)


#done with the principal component analysis, now moving on to clustering our data with EM algorithm.

#I am using the BIC score here to determine the number of clusters. I reused the code that I found in the
#scikit-learn documentation : https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html?fbclid=IwAR1efokdi6s7ssl3m6O2-_cnCMv5eeVelDg6N2mwUbIOIQ7EQnW2ei6j9eQ
#according to them, the lower the BIC score, the better the model. But if the number of clusters turns out to be a large one
#then we should consider the one, where the gradient is the highest.
#For this particular dataset, I found k=3 has the lowest BIC score, and we can also see that it has the greatest
#gradient, i.e. the greatest change in BIC scores. So, 3 would be the best possible value for the number of clusters.

def determine_number_of_clusters(samples):
    n_components_range = range(1, 7)
    bic = []

    lowest_bic = np.infty
    cv_types = ["full"]
    best_number_of_clusters = 0

    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_types[0]
        )
        gmm.fit(samples)
        bic.append(gmm.bic(samples))

        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            # best_gmm = gmm
            best_number_of_clusters = n_components

    # bic = np.array(bic)
    # print(bic)
    return best_number_of_clusters


K = determine_number_of_clusters(new_data)
print(K)


#we have got the perfect number of clusters, now we can move on to perform the EM algorithm

#implement EM algorithm

#1 - initialize parameters for K=3 clusters i.e. Gaussian Distribution

#priors (equal probability for each of the clusters)
w = []

for i in range(K):
    w.append(float(1/K))

w = np.array(w)


#centroids (means)
means = []
min_pc1 = new_data[0].min()
min_pc2 = new_data[1].min()

max_pc1 = new_data[0].max()
max_pc2 = new_data[1].max()

for i in range(K):
    x = np.random.uniform(min_pc1, max_pc1)
    y = np.random.uniform(min_pc2, max_pc2)

    means.append(np.array([x, y]))

means = np.array(means)


#covariance matrix
covariance = []

for i in range(K):
    covariance.append(np.array([[5.0, 0.0], [0.0, 5.0]]))

covariance = np.array(covariance)

print("Initial Prior : ", w)
print("Initial Mean : ", means)
print("Initial Covariance : ", covariance)


#our initialization has been done, now defining the function to calculate the likelihood for each sample,
#given a specific cluster/distribution

def compute_likelihood(current_observation, distribution_mean, distribution_covariance, dimension):
    deviation = current_observation - distribution_mean                      #dimension - dx1
    deviationT = deviation.transpose()                                       #dimension - 1xd
    inverse_of_covariance = np.linalg.inv(distribution_covariance)           #dimension - dxd
    first_portion = np.dot(deviationT, inverse_of_covariance)                #dimension - 1xd
    second_portion = np.dot(first_portion, deviation)                        #dimension - 1x1

    exp = math.exp(-0.5 * second_portion)
    probability = float(1 / (math.sqrt((2*3.1416)**dimension * np.linalg.det(distribution_covariance)))) * exp

    return probability


#now we should start the iterations; to calculate the posteriors, means, covariance matrix, priors, I followed the
#formulas that were mentioned in the assignment specification pdf file provided by my institution, I have attached the
#file in this project.

posteriors = []

for i in range(sample_size):
    posteriors.append(np.array([0.0, 0.0, 0.0]))

posteriors = np.array(posteriors)


log_likelihood = 0.0
number_of_iterations = 0
figure_number = 1

while True:

    if number_of_iterations % 5 == 0:
        draw_clusters(new_data, posteriors, figure_number)
        figure_number += 1

    #Expectation Step
    for i in range(sample_size):
        likelihoods = []
        denominator = 0.0
        for k in range(K):
            likelihood = compute_likelihood(new_data[i].reshape(2, 1), means[k].reshape(2, 1), covariance[k], 2)
            likelihoods.append(likelihood)
            denominator += w[k] * likelihood

        posteriors[i][0] = (w[0] * likelihoods[0]) / denominator
        posteriors[i][1] = (w[1] * likelihoods[1]) / denominator
        posteriors[i][2] = (w[2] * likelihoods[2]) / denominator


    #Maximization Step
    for k in range(K):
        numerator = 0.0
        denominator = 0.0

        for i in range(sample_size):
            numerator += posteriors[i][k] * new_data[i]
            denominator += posteriors[i][k]

        new_mean = numerator / denominator
        means[k] = new_mean

        # print("new mean : ", means)

        numerator = 0.0

        for i in range(sample_size):
            distance = new_data[i] - means[k]       #1xd
            distance = distance.reshape(2, 1)       #dx1
            distanceT = distance.transpose()        #1xd
            product = np.dot(distance, distanceT)   #dxd

            numerator += posteriors[i][k] * product

        new_covariance = numerator / denominator
        covariance[k] = new_covariance

        # print("new covariance : ", covariance)

        numerator = denominator
        w[k] = float(numerator / sample_size)

        # print("new prior : ", w)


    #Determine log-likelihood
    new_log_likelihood = 0.0

    for i in range(sample_size):
        sum_of_weighted_likelihoods = 0.0

        for k in range(K):
            likelihood = compute_likelihood(new_data[i].reshape(2, 1), means[k].reshape(2, 1), covariance[k], 2)
            sum_of_weighted_likelihoods += w[k] * likelihood

        log_sum = math.log(sum_of_weighted_likelihoods, 2)
        new_log_likelihood += log_sum


    if abs(new_log_likelihood - log_likelihood) < 0.000000000000000001:
        print("Converged! Number of iterations : ", number_of_iterations)
        print('\n')
        print("Prior : ", w)
        print("Mean : ", means)
        print("Covariance : ", covariance)

        break
    else:
        log_likelihood = new_log_likelihood
        print("iteration no. : " + str(number_of_iterations + 1) + " , current log-likelihood : ", log_likelihood)

    number_of_iterations += 1


#save the animation file
imageio.mimsave('animation.gif', images_for_animation, fps=2)


#show animation
file = "animation.gif"
animation = pyglet.resource.animation(file)
sprite = pyglet.sprite.Sprite(animation)
win = pyglet.window.Window(width=sprite.width, height=sprite.height)
green = 0, 1, 0, 1
pyglet.gl.glClearColor(*green)
@win.event
def on_draw():
    win.clear()
    sprite.draw()
pyglet.app.run()


#that's all! thanks for executing it!
