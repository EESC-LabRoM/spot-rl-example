import os
import xml.etree.ElementTree as ET


def parse_urdf_limits(urdf_path):
    """
    Parses a URDF file and returns a dictionary of joint limits.
    Returns: {joint_name: {'lower': float, 'upper': float, 'velocity': float}}
    """
    if not os.path.exists(urdf_path):
        print(f"Warning: URDF file {urdf_path} not found.")
        raise FileNotFoundError(f"URDF file {urdf_path} not found.")

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits = {}

    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        
        if joint.attrib.get("type") == "fixed":
            continue

        # validate limits exist        
        if limit is None: 
            raise ValueError(f"Joint {name} has no limit tag.")
        if 'lower' not in limit.attrib:
            raise ValueError(f"Joint {name} limit tag missing 'lower' attributes.")
        if 'upper' not in limit.attrib:
            raise ValueError(f"Joint {name} limit tag missing 'upper' attributes.")
        if 'velocity' not in limit.attrib:
            print(f"Warning: Joint {name} limit tag missing 'velocity' attribute. Setting to infinity.")

        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        velocity = float(limit.attrib.get("velocity", float("inf")))
        limits[name] = {'lower': lower, 'upper': upper, 'velocity': velocity}
    
    return limits
