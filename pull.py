import os
import json
from io import BytesIO

import mysql.connector
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box
import random


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'hotel-id',
    'password': 'TJ9dMJB^e%G9v$5a',
    'database': 'hotel-id',
    'port': 3307,
}

# Base path to save images and JSON
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "casino_floors")


def connect_to_database():
    """
    Establishes a connection to the MySQL database.
    """
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            port=DB_CONFIG['port']
        )
        return connection
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


def fetch_image_file(image_room_id):
    """
    Fetches the `file` field for a given `image_room.id` using a separate cursor.
    """
    # Open a new connection for this query
    connection = connect_to_database()
    if not connection:
        return None

    try:
        cursor = connection.cursor()
        query = "SELECT `file` FROM image_room WHERE id = %s"
        cursor.execute(query, (image_room_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    except mysql.connector.Error as e:
        print(f"Error fetching image file for image_room_id {image_room_id}: {e}")
        return None
    finally:
        # Close the secondary connection
        connection.close()



def process_snip_results(records):
    """
    Processes the query results, crops images, and saves them to the specified path.

    Also builds a hotel ID to name associative array.
    """
    hotels = {}
    last_image_room_id = None
    current_image_file = None

    # Iterate over each result in the query
    for record in records:
        city_id, hotel_id, hotel_name, room_id, image_room_id, snip_id, left, top, width, height = record

        # Fetch the image file only when image_room_id changes
        if last_image_room_id != image_room_id:
            current_image_file = fetch_image_file(image_room_id)
            last_image_room_id = image_room_id

        if current_image_file is None:
            print(f"Skipping processing for image_room_id {image_room_id} as no file was fetched.")
            continue

        # Construct paths
        ext = ".jpeg"  # Assuming JPEG format for all images
        image_filename = f"s{snip_id}{ext}"
        save_dir = os.path.join(BASE_PATH, str(city_id), str(hotel_id), str(room_id))
        os.makedirs(save_dir, exist_ok=True)  # Create directories if not exists
        save_path = os.path.join(save_dir, image_filename)

        # Update hotel associative array
        if hotel_id not in hotels:
            hotels[hotel_id] = hotel_name

        # Crop and save the image
        try:
            with Image.open(BytesIO(current_image_file)) as img:
                cropped_img = img.crop((left, top, left + width, top + height))
                cropped_img.save(save_path)
                print(f"Saved cropped image to {save_path}")
        except Exception as e:
            print(f"Error processing image for snip_id {snip_id}: {e}")

    return hotels

def doSnips():
    connection = connect_to_database()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = """
        SELECT city.id, hotel.id, hotel.name, room.id, image_room.id, snip.id, snip.left, snip.top, snip.width, snip.height
        FROM hotel
        INNER JOIN city ON city.id=hotel.city_id
        INNER JOIN room ON room.hotel_id=hotel.id
        INNER JOIN image_room ON image_room.room_id=room.id
        INNER JOIN snip ON snip.image_room_id=image_room.id
        WHERE room.room_type_id=2
        ORDER BY image_room.id
        """
        cursor.execute(query)

        records = []
        for record in cursor:
            records.append(record);
        cursor.close()
        connection.close()

        # Process the results and crop images
        hotels = process_snip_results(records)

        return hotels;
    except mysql.connector.Error as e:
        print(f"Error executing snip query: {e}")



def calculate_polygon_area(points):
    """
    Calculate the area of a polygon given a list of points [(x1,y1), (x2,y2), ...]
    using the Shoelace formula.
    """
    n = len(points)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    # Take absolute value and divide by 2
    area = abs(area) / 2.0

    return area



def has_significant_overlap(square, squares, threshold=0.5):
    """
    Check if the candidate square overlaps significantly with any existing squares.

    Args:
        square: List of (x,y) tuples representing the candidate square vertices
        squares: List of existing squares, each as a list of (x,y) tuples
        threshold: Minimum overlap ratio (default 0.9 for 90% overlap)

    Returns:
        Boolean: True if significant overlap exists, False otherwise
    """
    candidate = Polygon(square)
    candidate_area = candidate.area

    for existing_square in squares:
        existing_poly = Polygon(existing_square)

        # Calculate intersection area
        intersection_area = candidate.intersection(existing_poly).area

        # Calculate overlap ratio relative to candidate square area
        overlap_ratio = intersection_area / candidate_area

        if overlap_ratio >= threshold:
            return True

    return False

def generate_random_square_in_polygon(points_formatted, side, squares):
    """
    Generate coordinates for a square of side length SNIPSIDE that fits within the given polygon.

    Args:
        points_formatted: List of (x,y) tuples representing the polygon vertices
        SNIPSIDE: Length of the square's side

    Returns:
        List of (x,y) tuples representing the square vertices, or None if no solution found
    """
    # Create Shapely polygon from points
    polygon = Polygon(points_formatted)

    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Maximum number of attempts to find a valid square
    MAX_ATTEMPTS = 1000

    for _ in range(MAX_ATTEMPTS):
        # Generate random point within the polygon's bounding box
        # Adjust the range to account for square size
        px = random.uniform(minx, maxx - side)
        py = random.uniform(miny, maxy - side)

        # Create a square from this point
        square = box(px, py, px + side, py + side)

        # Check if square is completely within the polygon
        if polygon.contains(square) and not has_significant_overlap(square.exterior.coords, squares):
            # Return the coordinates of the square
            square_coords = list(square.exterior.coords)[:-1]  # Remove last point (same as first)
            return square_coords

    return None


SNIPSIDE = 100

def process_area_results(records):

    hotels = {}
    last_image_room_id = None
    current_image_file = None

    total_snips = 0

    # Iterate over each result in the query
    for record in records:
        city_id, hotel_id, hotel_name, room_id, image_room_id, area_id, coordinates = record

        points = json.loads(coordinates)
        points_formatted = [(p['x'], p['y']) for p in points]

        # Fetch the image file only when image_room_id changes
        if last_image_room_id != image_room_id:
            current_image_file = fetch_image_file(image_room_id)
            last_image_room_id = image_room_id

        if current_image_file is None:
            print(f"Skipping processing for image_room_id {image_room_id} as no file was fetched.")
            continue

        # Update hotel associative array
        if hotel_id not in hotels:
            hotels[hotel_id] = hotel_name



        # Crop and save the image
        try:
            with Image.open(BytesIO(current_image_file)) as img:
                draw = ImageDraw.Draw(img)
                polyColor = (0, 255, 0)
                squareColor = (0, 255, 255)
                #draw.polygon(points_formatted, outline=polyColor, fill=None)

                area = calculate_polygon_area(points_formatted)
                numSnips = 2 * int(area / (SNIPSIDE * SNIPSIDE))
                print(f"Number of potential snips: {numSnips}")

                squares = []
                for i in range(numSnips):
                    square = generate_random_square_in_polygon(points_formatted, SNIPSIDE, squares)
                    if square is not None:
                        squares.append(square)

                        # Construct paths
                        ext = ".jpeg"  # Assuming JPEG format for all images
                        image_filename = f"a{area_id}-{len(squares)}{ext}"
                        save_dir = os.path.join(BASE_PATH, str(city_id), str(hotel_id), str(room_id))
                        os.makedirs(save_dir, exist_ok=True)  # Create directories if not exists
                        save_path = os.path.join(save_dir, image_filename)

                        x_coords = [p[0] for p in square]
                        y_coords = [p[1] for p in square]

                        # Calculate the bounds
                        left = int(min(x_coords))
                        top = int(min(y_coords))
                        right = int(max(x_coords))
                        bottom = int(max(y_coords))

                        cropped_img = img.crop((left, top, right, bottom))
                        cropped_img.save(save_path)
                        #draw.polygon(square, outline=squareColor, fill=None)

                #img.show()
                # img.save(save_path)
                print(f"Number of actual snips: {len(squares)}")
                total_snips += len(squares)
                #print(f"Saved area image to {save_path}")
        except Exception as e:
            print(f"Error processing image for snip_id {area_id}: {e}")

    print(f"Total number of snips: {total_snips}")

    return hotels



def doAreas() :
    connection = connect_to_database()
    if not connection:
        return
    cursor = connection.cursor()
    query = """
    SELECT city.id, hotel.id, hotel.name, room.id, image_room.id, area.id, area.coordinates
    FROM hotel
    INNER JOIN city ON city.id=hotel.city_id
    INNER JOIN room ON room.hotel_id=hotel.id
    INNER JOIN image_room ON image_room.room_id=room.id
    INNER JOIN area ON area.image_room_id=image_room.id and area.image_type_id=1
    WHERE room.room_type_id=2
    ORDER BY image_room.id
    """
    cursor.execute(query)

    records = []
    for record in cursor:
        records.append(record);
    cursor.close()
    connection.close()

    # Process the results and crop images
    hotels = process_area_results(records)

    return hotels;


def main():

    doSnips()
    doAreas()


if __name__ == '__main__':
    main()
