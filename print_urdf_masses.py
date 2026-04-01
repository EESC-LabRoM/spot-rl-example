import xml.etree.ElementTree as ET
import sys

# Path to the URDF file (edit as needed)
urdf_path = "rl_deploy/spot/spot_with_arm.urdf"

def get_link_masses(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    link_masses = []
    for link in root.findall("link"):
        name = link.get("name")
        mass_elem = link.find("inertial/mass")
        if mass_elem is not None:
            mass = float(mass_elem.get("value"))
        else:
            mass = 0.0
        link_masses.append((name, mass))
    return link_masses

def main():
    urdf_file = sys.argv[1] if len(sys.argv) > 1 else urdf_path
    link_masses = get_link_masses(urdf_file)
    total_mass = 0.0
    print("Link masses:")
    for name, mass in link_masses:
        print(f"  {name}: {mass}")
        total_mass += mass
    print(f"Total mass: {total_mass}")

if __name__ == "__main__":
    main()
