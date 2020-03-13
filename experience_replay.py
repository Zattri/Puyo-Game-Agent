import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ExperienceReplay():
    observations = []

    def appendObservation(self, episode, step, info, action, reward, obs_img):
        # Need to buffer each observation into blocks of 4 to create an "Experience"
        compressed_img = self.compressObservation(obs_img)
        img_list = compressed_img.tolist()
        obs_tuple = (episode, step, info, action, reward, img_list)
        self.observations.append(obs_tuple)

    def saveFile(self, file_name="data"):
        file_path = file_name + ".json"
        print(f"Saving observations to {file_path}...")
        with open(file_path, "w") as fjson:
            json.dump(self.observations, fjson, cls=NumpyEncoder)
            fjson.close()
        print("Finished Saving!")

    def getObservation(self, x_index, y_index):
        return self.observations[x_index][y_index]

    def compressObservation(self, obs):
        # Do some compression on the numpy array here to downsample image size
        return obs

    #def readObservationsFromFile(self, file_path):
        #json_load = json.loads(json_dump)
        #a_restored = np.asarray(json_load["a"])
