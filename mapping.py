import os
import cv2
import json
from typing import Union
import numpy as np

ROI_PATH = r"roi_data"


# type is "map" or "cam"
def load_roi_data(type: list["map", "camera"]) -> dict:
    data = {}
    specific_roi_path = os.path.join(ROI_PATH, type)
    for json_file in os.listdir(specific_roi_path):
        if json_file.endswith(".json"):
            with open(os.path.join(specific_roi_path, json_file), 'r') as f:
                temp = json.load(f)
            for key_roi_name in temp["roi"]:
                temp_point_list = []
                for key_roi_point in temp["roi"][key_roi_name]:
                    temp_point_list.append(temp["roi"][key_roi_name][key_roi_point])
                    
                temp_point_list = np.array(temp_point_list, np.float32)
                temp["roi"][key_roi_name] = temp_point_list

            data[json_file.split(".")[0]] = temp

    return data


def find_perspective_transform(map_roi, cam_roi):
    return cv2.getPerspectiveTransform(map_roi, cam_roi)

def write_json_file(data, file_path):
    for H_key, H_value in data.items():
        if isinstance(H_value, np.ndarray):
            data[H_key] = H_value.tolist()
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    map_dict= load_roi_data(type="map")
    cam_dict = load_roi_data(type="camera")


    matrix_H = {}
    for map_name in map_dict: # map level
        map_roi = map_dict[map_name]["roi"]
        for map_roi_name, map_roi_data in map_roi.items(): 

            for cam_name in cam_dict:
                cam_roi = cam_dict[cam_name]["roi"]
                for cam_roi_name, cam_roi_data in cam_roi.items():
                    
                    if cam_roi_name == map_roi_name:
                        transform_matrix = find_perspective_transform(cam_roi_data, map_roi_data)
                        matrix_H[cam_roi_name] = transform_matrix
                        print(f"Transform matrix for {cam_roi_name}:")
                        print(cam_roi_data)
                        print(f"Map ROI data for {map_roi_name}:")
                        print(map_roi_data)
                        print(f"Transform matrix for {cam_roi_name} to {map_roi_name}:")
                        print(transform_matrix)

    write_json_file(matrix_H, "H_cam2map.json")
    
    # print(map_data)
