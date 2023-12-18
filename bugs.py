import tkinter as tk
from tkinter import PhotoImage, Toplevel
from PIL import Image, ImageTk, ImageGrab
import csv, os
from datetime import datetime

def on_drag(event):
    global currently_dragged_symbol

    if currently_dragged_symbol is None:
        # Determine which symbol is being dragged
        item = canvas.find_closest(event.x, event.y)[0]
        currently_dragged_symbol = next((s for s in symbols if s['image_id'] == item), None)

    if currently_dragged_symbol is None:
        return

    # Calculate the offset from the symbol's center
    offset_x = event.x - currently_dragged_symbol['coords'][0]
    offset_y = event.y - currently_dragged_symbol['coords'][1]

    # Update the symbol's coordinates
    new_x = currently_dragged_symbol['coords'][0] + offset_x
    new_y = currently_dragged_symbol['coords'][1] + offset_y

    # Get canvas size
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Ensure the symbol stays within the boundaries
    x = min(max(new_x, 0), canvas_width)
    y = min(max(new_y, 0), canvas_height)

    # Update the position of the symbol on the canvas
    canvas.coords(currently_dragged_symbol['image_id'], x, y)

    # Update the coordinates in the symbol dictionary
    currently_dragged_symbol['coords'] = (x, y)

def stop_drag(event):
    global currently_dragged_symbol
    currently_dragged_symbol = None


def resize_image(image_path, scale_factor):
    # Open the image file
    img = Image.open(image_path)
    # Calculate the new size
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # Resize the image
    resized_img = img.resize((new_width, new_height))
    return ImageTk.PhotoImage(resized_img)

def submit_name():
    global user_name
    user_name = name_entry.get()
    if user_name:
        print(f"User Name: {user_name}")
        overlay_frame.lower(canvas)  # Lower the overlay behind the canvas
    else:
        print("Please enter your name.")

def open_scary_rating_window(symbol):
    # Create a new top-level window for rating
    rating_window = Toplevel(root, bg='lightblue')
    rating_window.title(f"How Scary is this Bug?")

    # Display the symbol image
    symbol_image = ImageTk.PhotoImage(Image.open(symbol['image_path']))  # Resize as needed
    symbol_label = tk.Label(rating_window, image=symbol_image)
    symbol_label.image = symbol_image  # Keep a reference to prevent garbage collection
    symbol_label.pack()

    # Display text below the symbol
    text_label = tk.Label(rating_window, text=f"How scary is this bug?\n1 for not scary, 7 for very scary", font=('Menlo', 12))
    text_label.pack()

    # Function to handle rating button click
    def on_rating_button_click(rating):
        global rated_symbols_count
        # Log the rating to a CSV file
        with open("ratings.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([symbol['id'], rating, "scary", user_name])
        print(f"Symbol {symbol['id']} rated as: {rating}")
        rating_window.destroy()
        
        rated_symbols_count += 1
        if rated_symbols_count == len(scary_bugs_ids + cute_bugs_ids):
            root.quit()

    # Create buttons for rating from 1 to 7
    for i in range(1, 8):
        btn = tk.Button(rating_window, text=str(i), command=lambda i=i: on_rating_button_click(i))
        btn.pack(side=tk.LEFT)

def open_cute_rating_window(symbol):
    # Create a new top-level window for rating
    rating_window = Toplevel(root, bg='lightpink')
    rating_window.title(f"How Cute is this Bug?")

    # Display the symbol image
    symbol_image = ImageTk.PhotoImage(Image.open(symbol['image_path']))  # Resize as needed
    symbol_label = tk.Label(rating_window, image=symbol_image)
    symbol_label.image = symbol_image  # Keep a reference to prevent garbage collection
    symbol_label.pack()

    # Display text below the symbol
    text_label = tk.Label(rating_window, text=f"How cute is this bug?\n1 for not cute, 7 for very cute", font=('Menlo', 12))
    text_label.pack()

    # Function to handle rating button click
    def on_rating_button_click(rating):
        global rated_symbols_count
        # Log the rating to a CSV file
        with open("ratings.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([symbol['id'], rating, "cute", user_name])
        print(f"Symbol {symbol['id']} rated as: {rating}")
        rating_window.destroy()
        
        rated_symbols_count += 1
        if rated_symbols_count == len(scary_bugs_ids + cute_bugs_ids):
            root.quit()

    # Create buttons for rating from 1 to 7
    for i in range(1, 8):
        btn = tk.Button(rating_window, text=str(i), command=lambda i=i: on_rating_button_click(i))
        btn.pack(side=tk.LEFT)

def save_coordinates():
    # Define the file path
    file_path = "coordinates_log.csv"

    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save the canvas as an image
    canvas_image = ImageGrab.grab(bbox=(canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_rootx() + canvas.winfo_width(), canvas.winfo_rooty() + canvas.winfo_height()))
    canvas_image.save(f"canvas_{user_name}.png")

    # Save the coordinates to a CSV file
    with open(file_path, "a", newline='') as file:
        writer = csv.writer(file)

        # Write headers if the file is being created
        if not file_exists:
            writer.writerow(["Time", "Username", "Symbol ID", "Coordinates"])

        # Write data rows
        for symbol in symbols:
            writer.writerow([current_time, user_name, symbol['id'], f"({symbol['coords'][0]}, {symbol['coords'][1]})"])

    print("Coordinates saved to 'coordinates_log.csv'")

    # Open rating windows only for the specified subset of symbols

    for symbol in symbols:
        if symbol['id'] in scary_bugs_ids:
            open_scary_rating_window(symbol)
    for symbol in symbols:
        if symbol['id'] in cute_bugs_ids:
            open_cute_rating_window(symbol)

################### main #####################

root = tk.Tk()
root.title("Group the beetles!")

# Overlay frame for user input
overlay_frame = tk.Frame(root, bg='white')
overlay_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)

# Label for instructions
instructions = "Please drag beetles into groupings of your choosing, press save clusters when complete\nEnter your name to begin."
instruction_label = tk.Label(overlay_frame, text=instructions, font=('Menlo', 20), bg='white')
instruction_label.pack(pady=(20, 10))

# Entry widget for name input
name_entry = tk.Entry(overlay_frame, font=('Arial', 20))
name_entry.pack(pady=20)

# Submit button
submit_button = tk.Button(overlay_frame, text="OK", command=submit_name, font=('Arial', 20))
submit_button.pack(pady=10)

# Create a canvas to draw the symbols
canvas = tk.Canvas(root, width=1200, height=800)
canvas.pack()

symbols = [
    {'id': 1, 'coords': (125, 150), 'image_path': 'index/01.png'},
    {'id': 2, 'coords': (250, 150), 'image_path': 'index/02.png'},
    {'id': 3, 'coords': (375, 150), 'image_path': 'index/03.png'},
    {'id': 4, 'coords': (500, 150), 'image_path': 'index/04.png'},
    {'id': 5, 'coords': (625, 150), 'image_path': 'index/05.png'},
    {'id': 6, 'coords': (750, 150), 'image_path': 'index/06.png'},
    {'id': 7, 'coords': (875, 150), 'image_path': 'index/07.png'},
    {'id': 8, 'coords': (1000, 150), 'image_path': 'index/08.png'},
    {'id': 9, 'coords': (125, 300), 'image_path': 'index/09.png'},
    {'id': 10, 'coords': (250, 300), 'image_path': 'index/10.png'},
    {'id': 11, 'coords': (375, 300), 'image_path': 'index/11.png'},
    {'id': 12, 'coords': (500, 300), 'image_path': 'index/12.png'},
    {'id': 13, 'coords': (625, 300), 'image_path': 'index/13.png'},
    {'id': 14, 'coords': (750, 300), 'image_path': 'index/14.png'},
    {'id': 15, 'coords': (875, 300), 'image_path': 'index/15.png'},
    {'id': 16, 'coords': (1000, 300), 'image_path': 'index/16.png'},
    {'id': 17, 'coords': (125, 450), 'image_path': 'index/17.png'},
    {'id': 18, 'coords': (250, 450), 'image_path': 'index/18.png'},
    {'id': 19, 'coords': (375, 450), 'image_path': 'index/19.png'},
    {'id': 20, 'coords': (500, 450), 'image_path': 'index/20.png'},
    {'id': 21, 'coords': (625, 450), 'image_path': 'index/21.png'},
    {'id': 22, 'coords': (750, 450), 'image_path': 'index/22.png'},
    {'id': 23, 'coords': (875, 450), 'image_path': 'index/23.png'},
    {'id': 24, 'coords': (1000, 450), 'image_path': 'index/24.png'},
    {'id': 25, 'coords': (125, 600), 'image_path': 'index/25.png'},
    {'id': 26, 'coords': (250, 600), 'image_path': 'index/26.png'},
    {'id': 27, 'coords': (375, 600), 'image_path': 'index/27.png'},
    {'id': 28, 'coords': (500, 600), 'image_path': 'index/28.png'},
    {'id': 29, 'coords': (625, 600), 'image_path': 'index/29.png'},
    {'id': 30, 'coords': (750, 600), 'image_path': 'index/30.png'},
    {'id': 31, 'coords': (875, 600), 'image_path': 'index/31.png'},
    {'id': 32, 'coords': (1000, 600), 'image_path': 'index/32.png'}
]


scary_bugs_ids = [1, 8, 19, 23, 12, 15]
cute_bugs_ids = [32, 10, 16, 31, 6, 14]

# Desired size for the images
scale_factor = 0.25  # Adjust the size as needed

# Load images, resize them, and create canvas image items
for symbol in symbols:
    img = resize_image(symbol['image_path'], scale_factor)
    symbol['image'] = img  # Store the resized PhotoImage object to prevent garbage collection
    symbol['image_id'] = canvas.create_image(symbol['coords'], image=img)

# Bind the dragging action to each symbol
for symbol in symbols:
    canvas.tag_bind(symbol['image_id'], '<B1-Motion>', on_drag)

# Create a button to save coordinates
save_button = tk.Button(root, text="Save Clusters", command=save_coordinates, font=('Menlo', 20), padx=20, pady=10)
save_button.pack()

canvas.bind('<ButtonRelease-1>', stop_drag)

# Global variables
rated_symbols_count = 0
currently_dragged_symbol = None

overlay_frame.lift()  # Make the overlay frame appear on top of the canvas
root.mainloop()