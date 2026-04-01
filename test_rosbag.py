"""
Script to list all topics in a ROS 2 bag using the rosbags Python package.
"""

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores
from pathlib import Path

# Path to the ROS 2 bag directory (not the parent folder, but the folder containing metadata.yaml)
bag_path = Path('rosbag2_2025_09_01-19_17_41-20260326T174257Z-1-001/rosbag2_2025_09_01-19_17_41')

# Open the bag using AnyReader with default_typestore
with AnyReader([bag_path], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
    # Print all topics in the bag
    print('Topics in the bag:')
    for topic in reader.topics:
        print(topic, reader.topics[topic].msgtype)

    # Print the first message from /joint_states
    joint_states_topic = '/joint_states'
    print(f'\nFirst message from {joint_states_topic}:')
    # Find all connections for the topic
    connections = [c for c in reader.connections if c.topic == joint_states_topic]
    found = False
    for connection, timestamp, rawdata in reader.messages(connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        print(msg)
        found = True
        break
    if not found:
        print('No messages found for this topic.')
