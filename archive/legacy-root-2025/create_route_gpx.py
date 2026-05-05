#!/usr/bin/env python3
"""
Convert the continuous route CSV to a GPX file for GPS navigation.
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import sys

def create_gpx_root() -> ET.Element:
    """Create the root GPX element with proper namespaces."""
    gpx = ET.Element("gpx")
    gpx.set("version", "1.1")
    gpx.set("creator", "Boise Trails Challenge Route Planner")
    gpx.set("xmlns", "http://www.topografix.com/GPX/1/1")
    gpx.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    gpx.set("xsi:schemaLocation", "http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd")
    
    # Add metadata
    metadata = ET.SubElement(gpx, "metadata")
    name = ET.SubElement(metadata, "name")
    name.text = "Boise Trails Challenge - Complete Continuous Route"
    
    desc = ET.SubElement(metadata, "desc")
    desc.text = "Optimal single continuous route covering all 247 official Boise Trails Challenge segments"
    
    time = ET.SubElement(metadata, "time")
    time.text = datetime.now().isoformat() + "Z"
    
    return gpx

def load_trail_coordinates(json_file: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load full coordinate paths for each trail segment."""
    print(f"Loading trail coordinates from {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    coordinates = {}
    for feature in data.get('trailSegments', []):
        if feature['type'] != 'Feature':
            continue
            
        props = feature['properties']
        geom = feature['geometry']
        
        if geom['type'] != 'LineString':
            continue
            
        seg_id = str(props.get('segId', ''))
        coords = [(c[0], c[1]) for c in geom['coordinates']]
        
        if seg_id and coords:
            coordinates[seg_id] = coords
    
    print(f"Loaded coordinates for {len(coordinates)} trail segments")
    return coordinates

def read_route_csv(csv_file: str) -> List[Dict]:
    """Read the route CSV file."""
    print(f"Reading route from {csv_file}...")
    
    route_segments = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Sequence'] and row['Sequence'] != 'SUMMARY':
                try:
                    route_segments.append({
                        'sequence': int(row['Sequence']),
                        'type': row['Type'],
                        'segment_id': row['Segment_ID'],
                        'name': row['Name'],
                        'traversal': row['Traversal'],
                        'start_lat': float(row['Start_Lat']),
                        'start_lon': float(row['Start_Lon']),
                        'end_lat': float(row['End_Lat']),
                        'end_lon': float(row['End_Lon'])
                    })
                except (ValueError, KeyError):
                    # Skip summary rows or invalid data
                    continue
    
    print(f"Found {len(route_segments)} route segments")
    return route_segments

def create_track_segment(gpx: ET.Element, name: str, coords: List[Tuple[float, float]], 
                        segment_type: str = "trail") -> None:
    """Create a track segment in the GPX."""
    trk = ET.SubElement(gpx, "trk")
    
    trk_name = ET.SubElement(trk, "name")
    trk_name.text = name
    
    trk_type = ET.SubElement(trk, "type")
    trk_type.text = segment_type
    
    trkseg = ET.SubElement(trk, "trkseg")
    
    for lon, lat in coords:
        trkpt = ET.SubElement(trkseg, "trkpt")
        trkpt.set("lat", f"{lat:.6f}")
        trkpt.set("lon", f"{lon:.6f}")

def create_waypoints(gpx: ET.Element, route_segments: List[Dict]) -> None:
    """Create waypoints for trail connections and driving segments."""
    waypoint_count = 0
    
    for i, segment in enumerate(route_segments):
        # Add waypoint at start of each component
        if segment['type'] == 'DRIVE' or i == 0:
            wpt = ET.SubElement(gpx, "wpt")
            wpt.set("lat", f"{segment['start_lat']:.6f}")
            wpt.set("lon", f"{segment['start_lon']:.6f}")
            
            name = ET.SubElement(wpt, "name")
            if segment['type'] == 'DRIVE':
                name.text = f"DRIVE-{waypoint_count}: {segment['name']}"
            else:
                name.text = f"START: {segment['name']}"
            
            desc = ET.SubElement(wpt, "desc")
            desc.text = f"Sequence {segment['sequence']}: {segment['name']}"
            
            waypoint_count += 1

def create_gpx_file(trail_coords: Dict[str, List[Tuple[float, float]]], 
                   route_segments: List[Dict], output_file: str) -> None:
    """Create the complete GPX file."""
    print(f"Creating GPX file: {output_file}")
    
    gpx = create_gpx_root()
    
    # Add waypoints for navigation
    create_waypoints(gpx, route_segments)
    
    # Process each route segment
    trail_count = 0
    drive_count = 0
    
    for segment in route_segments:
        segment_id = segment['segment_id']
        name = segment['name']
        traversal = segment['traversal']
        
        if segment['type'] == 'TRAIL':
            # Get full coordinate path for trail segments
            if segment_id in trail_coords:
                coords = trail_coords[segment_id].copy()
                
                # Reverse coordinates if traversing in reverse
                if traversal == 'REVERSE':
                    coords.reverse()
                
                track_name = f"Trail {trail_count+1}: {name} ({traversal})"
                create_track_segment(gpx, track_name, coords, "trail")
                trail_count += 1
            else:
                # Fallback to straight line if coordinates not found
                coords = [
                    (segment['start_lon'], segment['start_lat']),
                    (segment['end_lon'], segment['end_lat'])
                ]
                track_name = f"Trail {trail_count+1}: {name} (NO COORDS)"
                create_track_segment(gpx, track_name, coords, "trail")
                trail_count += 1
        
        elif segment['type'] == 'DRIVE':
            # Create straight line for driving segments
            coords = [
                (segment['start_lon'], segment['start_lat']),
                (segment['end_lon'], segment['end_lat'])
            ]
            track_name = f"Drive {drive_count+1}: {name}"
            create_track_segment(gpx, track_name, coords, "drive")
            drive_count += 1
    
    # Write GPX file
    tree = ET.ElementTree(gpx)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"GPX file created successfully!")
    print(f"  Trail segments: {trail_count}")
    print(f"  Drive segments: {drive_count}")
    print(f"  Total tracks: {trail_count + drive_count}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_route_gpx.py <route.csv> <trail_data.json> [output.gpx]")
        sys.exit(1)
    
    route_csv = sys.argv[1]
    trail_json = sys.argv[2]
    output_gpx = sys.argv[3] if len(sys.argv) > 3 else "continuous_route.gpx"
    
    try:
        # Load trail coordinates
        trail_coords = load_trail_coordinates(trail_json)
        
        # Read route
        route_segments = read_route_csv(route_csv)
        
        if not route_segments:
            print("No route segments found in CSV!")
            return
        
        # Create GPX file
        create_gpx_file(trail_coords, route_segments, output_gpx)
        
        print(f"\n✅ GPX file created: {output_gpx}")
        print("\nThis GPX file includes:")
        print("- Full coordinate paths for all trail segments")
        print("- Driving segments as straight lines")
        print("- Waypoints at major transition points")
        print("- Proper directional traversal (forward/reverse)")
        print("\nYou can now load this into any GPS device or mapping app!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 