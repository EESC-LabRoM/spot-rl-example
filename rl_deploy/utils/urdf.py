import os 
import xml.etree.ElementTree as ET
import numpy as np

def parse_urdf_limits(urdf_path):
    """
    Parses a URDF file and returns a dictionary of joint limits.
    Returns: {joint_name: {'lower': float, 'upper': float, 'velocity': float}}
    """
    if not os.path.exists(urdf_path):
        print(f"Warning: URDF file {urdf_path} not found.")
        return {}

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        limits = {}

        for joint in root.findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")

            # validate limits exist        
            if limit is None: 
                raise ValueError(f"Joint {name} has no limit tag.")
            if 'lower' not in limit.attrib:
                raise ValueError(f"Joint {name} limit tag missing 'lower' attributes.")
            if 'upper' not in limit.attrib:
                raise ValueError(f"Joint {name} limit tag missing 'upper' attributes.")
            if 'velocity' not in limit.attrib:
                print(f"Warning: Joint {name} limit tag missing 'velocity' attribute. Setting to infinity.")

            lower = float(limit["lower"])
            upper = float(limit["upper"])
            velocity = float(limit["velocity"])
            limits[name] = {'lower': lower, 'upper': upper, 'velocity': velocity}
        
        return limits
    except Exception as e:
        print(f"Error parsing URDF: {e}")
        return {}

