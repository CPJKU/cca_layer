
import os
import pickle
import numpy as np
from cca_layer.config.settings import DATA_ROOT


class MultiViewDataPool(object):
    """
    Data pool for multi-view observations
    """

    def __init__(self, observations_view1, observations_view2, shuffle=True):
        """ Constructor """

        self.observations_view1 = observations_view1
        self.observations_view2 = observations_view2

        self.dim_view1 = list(self.observations_view1[0].shape)
        self.dim_view2 = list(observations_view2[0].shape)

        self.shape = None

        # prepare train data
        self.train_entities = None
        self.prepare_train_entities()

        # shuffle data
        if shuffle:
            self.reset_batch_generator()

    def prepare_train_entities(self):
        """ collect train entities """

        self.train_entities = np.zeros((0, 2), dtype=np.int)

        # collect train entities
        # (this has to be changed if there is no clear 1:1 correspondence between the two views)
        # (e.g. one image that has five different captions to train on)
        for i_view1 in xrange(self.observations_view1.shape[0]):
            cur_entities = np.asarray([i_view1, i_view1])
            self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self, indices=None):
        """ reset batch generator """
        if indices is None:
            indices = np.random.permutation(self.shape[0])
        self.train_entities = self.train_entities[indices]

    def __getitem__(self, key):
        """ make class accessible by index or slice """

        # get batch
        if key.__class__ == int:
            key = slice(key, key + 1)
        batch_entities = self.train_entities[key]

        # collect train entities
        view1_batch = np.zeros(tuple([len(batch_entities)] + self.dim_view1), dtype=np.float32)
        view2_batch = np.zeros(tuple([len(batch_entities)] + self.dim_view2), dtype=np.float32)
        for i_entity, (i_v1, i_v2) in enumerate(batch_entities):

            # get current observations
            x = self.observations_view1[i_v1]
            y = self.observations_view2[i_v2]

            # collect batch data
            view1_batch[i_entity] = x
            view2_batch[i_entity] = y

        return view1_batch, view2_batch


def load_iapr(img_file="iapr_images_vgg_vectors.npy", cap_file="iapr_captions_tfidf.pkl", normalize_img=True, normalize_txt=True, seed=23):
    """
    Load IAPR features and word representations
    """
    np.random.seed(seed)

    # load images
    Images = np.load(os.path.join(DATA_ROOT, img_file))

    # load caption vectors
    with open(os.path.join(DATA_ROOT, cap_file), "rb") as fp:
        vector_captions, captions = pickle.load(fp)
        vector_captions = vector_captions[:, 0, :]

    # normalize images
    print "Images.max()", Images.max()
    if normalize_img:
        Images = Images.astype(np.float32)
        Images /= Images.max()

    # normalize vector space
    if normalize_txt:
        vector_captions -= vector_captions.min()
        vector_captions /= vector_captions.max()

    # split images into original and flipped version
    Images = Images[0:19996]

    # shuffle the data
    rand_idx = np.random.permutation(len(vector_captions))
    Images = Images[rand_idx]
    vector_captions = vector_captions[rand_idx]

    tr_images = Images[3000:]
    va_images = Images[0:1000]
    te_images = Images[1000:3000]

    tr_captions = vector_captions[3000:]
    va_captions = vector_captions[0:1000]
    te_captions = vector_captions[1000:3000]

    # reuse some images to have 17000 samples available
    tr_images = np.concatenate([tr_images, tr_images[0:4]])
    tr_captions = np.concatenate([tr_captions, tr_captions[0:4]])

    # initialize data pools
    train_pool = MultiViewDataPool(tr_images, tr_captions)
    valid_pool = MultiViewDataPool(va_images, va_captions, shuffle=False)
    test_pool = MultiViewDataPool(te_images, te_captions, shuffle=False)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


def load_audio_score(seed=23):
    """ Load audio score retrieval data """
    np.random.seed(seed)

    data = np.load(os.path.join(DATA_ROOT, "audio_sheet_music.npz"))

    # initialize data pools
    train_pool = MultiViewDataPool(data["X_tr"], data["Y_tr"], shuffle=True)
    valid_pool = MultiViewDataPool(data["X_va"], data["Y_va"], shuffle=False)
    test_pool = MultiViewDataPool(data["X_te"], data["Y_te"], shuffle=False)

    print("Train: %d" % train_pool.shape[0])
    print("Valid: %d" % valid_pool.shape[0])
    print("Test: %d" % test_pool.shape[0])

    return dict(train=train_pool, valid=valid_pool, test=test_pool)


if __name__ == "__main__":
    """ main """
    pass
