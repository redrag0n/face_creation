import time
import numpy as np

import GPy
import GPyOpt

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

#from IPython.display import clear_output



class BayessianOptimization:
    def __init__(self, model, latent_size):
        self.latent_size = latent_size
        self.model = model
        self.samples = None
        self.images = None
        self.rating = None
        self._labels_encoded = None

    def _get_image(self, latent, label):
        img = self.model.generate_from_t_sampled(latent, label)
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def _show_images(images, titles):
        assert len(images) == len(titles)
        #clear_output()
        plt.figure(figsize=(3 * len(images), 3))
        n = len(titles)
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(np.moveaxis(images[i], 0, -1))
            plt.title(str(titles[i]))
            plt.axis('off')
        plt.show()

    @staticmethod
    def _draw_border(image, w=2):
        bordred_image = image.copy()
        bordred_image[:, :w] = [1, 0, 0]
        bordred_image[:, -w:] = [1, 0, 0]
        bordred_image[:w, :] = [1, 0, 0]
        bordred_image[-w:, :] = [1, 0, 0]
        return bordred_image

    def get_initial_samples(self, n_start, select_top):
        '''
        Generate n_start samples, cluster them, and choose select_top cluster centroids
        '''
        initial_samples = np.random.randn(n_start, self.latent_size)
        # cls = KMeans(select_top)
        # cls.fit(initial_samples)
        distances = euclidean_distances(initial_samples)
        sample = initial_samples[0]
        new_samples = [sample]
        for i in range(select_top - 1):
            sample_i = np.argmax(distances[i])
            sample = initial_samples[sample_i]
            new_samples.append(sample)
            distances[:, sample_i] = 0
        # centroids = cls.cluster_centers_
        samples = np.stack(new_samples)
        return samples

    def query_initial(self, n_start=1000, select_top=5):
        '''
        Creates initial points for Bayesian optimization
        Generate *n_start* random images and asks user to rank them.
        Gives maximum score to the best image and minimum to the worst.
        :param n_start: number of images to rank initialy.
        :param select_top: number of images to keep
        '''
        images = list()
        ratings = list()
        self.samples = self.get_initial_samples(n_start, select_top)
        for i in range(select_top):
            sample = self.samples[i:i+1]
            images.append(self._get_image(sample, self._labels_encoded)[0])
            ratings.append(0)
        self.images = np.stack(images)
        self.rating = np.stack(ratings)

        # self.samples = np.random.randn(select_top, latent_size) ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        # self.images = np.clip(sess.run(decode, feed_dict={latent_placeholder: self.samples}), 0, 1) ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        # self.rating = np.zeros(select_top)

        ### YOUR CODE:
        ### Show user some samples (hint: use self._get_image and input())
        self._show_images(self.images, self.rating)
        print('Score images with numbers from 1 to 10. Input scores with spaces between them', flush=True)

        good_input = False
        # while not good_input:
        time.sleep(1)
        rating_str = input()
        ratings = [-int(score) for score in rating_str.split() if len(score) != 0]
        if len(ratings) == select_top:
            self.rating = np.array(ratings)

        # # Check that tensor sizes are correct
        # np.testing.assert_equal(self.rating.shape, [select_top])
        # np.testing.assert_equal(self.images.shape, [select_top, 64, 64, 3])
        # np.testing.assert_equal(self.samples.shape, [select_top, self.latent_size])

    def evaluate(self, candidate):
        '''
        Queries candidate vs known image set.
        Adds candidate into images pool.
        :param candidate: latent vector of size 1xlatent_size
        '''
        initial_size = len(self.images)

        ### YOUR CODE HERE
        ## Show user an image and ask to assign score to it.
        ## You may want to show some images to user along with their scores
        ## You should also save candidate, corresponding image and rating
        image = self._get_image(candidate[0:1], self._labels_encoded)[0]
        order = np.argsort(self.rating)
        ordered_images = self.images[order]
        # print(ordered_images.shape)
        image_to_show = np.vstack([ordered_images, np.expand_dims(image, axis=0)])
        self._show_images(image_to_show, list([-rating for rating in self.rating[order]]) + ['Score this image'])
        # self._show_images([image], [''])
        print('Score image')
        time.sleep(1)
        score = input()
        if '.' in score:
            candidate_rating = float(score)
        else:
            candidate_rating = int(score)
        candidate_rating = - candidate_rating
        # print('samples', self.samples.shape, candidate.shape)
        self.samples = np.vstack([self.samples, candidate])
        # print('images', self.images.shape, image.shape)
        self.images = np.vstack([self.images, np.expand_dims(image, axis=0)])
        # print('rating', self.rating.shape)
        self.rating = np.hstack([self.rating, candidate_rating])

        assert len(self.images) == initial_size + 1
        assert len(self.rating) == initial_size + 1
        assert len(self.samples) == initial_size + 1
        return candidate_rating

    def optimize(self, labels_encoded, n_iter=10, w=3, acquisition_type='EI', acquisition_par=0.15):
        self._labels_encoded = labels_encoded
        if self.samples is None:
            self.query_initial()

        bounds = [{'name': 'z_{0:03d}'.format(i),
                   'type': 'continuous',
                   'domain': (-w, w)}
                  for i in range(self.latent_size)]
        optimizer = GPyOpt.methods.BayesianOptimization(f=self.evaluate, domain=bounds,
                                                        acquisition_type=acquisition_type,
                                                        acquisition_jitter=acquisition_par,
                                                        exact_eval=False,  # Since we are not sure
                                                        model_type='GP',
                                                        X=self.samples,
                                                        Y=self.rating[:, None],
                                                        maximize=False)
        optimizer.run_optimization(max_iter=n_iter, eps=-1)

    def get_best(self):
        index_best = np.argmin(self.rating)
        return self.images[index_best]

    def draw_best(self, title=''):
        index_best = np.argmin(self.rating)
        image = self.images[index_best]
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()