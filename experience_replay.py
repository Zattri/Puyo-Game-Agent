import json
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ExperienceReplay():
    observations = []

    def appendObservation(self, episode, step, info, action, reward, obs_img):
        compressed_img = self.compressObservation(obs_img)
        #TODO: Do toList only when you want to save the data, not now - Maybe experiment with compressing later on, storing all screenshots in ram
        obs_tuple = (episode, step, info, action, reward, compressed_img.tolist())
        self.observations.append(obs_tuple)

    def saveFile(self, file_name="data"):
        file_path = file_name + ".json"
        print(f"Saving observations to {file_path}...")
        with open(file_path, "w") as fjson:
            json.dump(self.observations, fjson, cls=NumpyEncoder)
            fjson.close()
        print("Finished Saving!")

    def getObservation(self, x_index, y_index):
        return np.asarray(self.observations[x_index][y_index])

    def compressObservation(self, obs):
        return skimage.measure.block_reduce(obs, (2, 2, 1), np.max)

    #def readObservationsFromFile(self, file_path):
        #json_load = json.loads(json_dump)
        #a_restored = np.asarray(json_load["a"])