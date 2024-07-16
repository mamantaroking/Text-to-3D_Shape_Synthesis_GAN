import numpy as np
import torch
from scipy.linalg import sqrtm

# from text_trainer import fake_noise
# from new_texter_model3 import textD
import binvox_rw

def calculate_fid(real_features, fake_features):
    mu_real = np.mean(real_features, axis=0)
    # print(mu_real)
    mu_fake = np.mean(fake_features, axis=0)
    # print(mu_fake)
    cov_real = np.cov(real_features, rowvar=False)
    cov_fake = np.cov(fake_features, rowvar=False)
    # print(cov_real, cov_fake)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu_real-mu_fake)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(cov_real.dot(cov_fake))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # diff = mu_real - mu_fake
    # print(diff)
    # trace_term = np.trace(cov_real + cov_fake - 2 * np.sqrt(np.dot(cov_real, cov_fake)))

    # fid_score = np.dot(diff, diff) + trace_term
    fid_score = ssdiff + np.trace(cov_real + cov_fake - 2.0 * covmean)
    fid_score = max(fid_score, 0.0)
    return fid_score


'''def output():
    real_features = textD.main(real_data).detach().cpu().numpy()
    gen_features = textD.main(generated_data).detach().cpu().numpy()'''


# Example usage:
real_features = np.random.rand(1024, 512)  # Real features (extracted from real data)
fake_features = np.random.rand(1024, 512)  # Generated features (extracted from generated data)
fid = calculate_fid(real_features, fake_features)
# print(fid)


import numpy as np

# Your original array
original_array = np.random.rand(16, 512, 4, 4, 4)

# Reshape to [64, 512, 64]
# reshaped_array = original_array.reshape(1024, 512)
reshaped_array = original_array.reshape(16, 512, -1)
# reshaped_array = np.ravel(original_array)
# reshaped_array = np.squeeze(original_array)
# reshaped_array = original_array.squeeze()

# Verify the new shape
# print(reshaped_array.shape)  # Should be (64, 512, 64)

array = torch.randn([64, 1, 2, 2, 2])
array = array.squeeze()
# print(array.shape)