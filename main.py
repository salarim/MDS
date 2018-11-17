import subprocess
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

def get_random_polygon(radius, max_size):
    compile_proc = subprocess.Popen(["g++", "random_polygon.cpp", "-o", "random_poly.out", "-lCGAL", "-lgmp"], stdout=subprocess.PIPE)
    compile_out = compile_proc.communicate()
    if compile_out[0] or compile_out[1]:
        print(compile_out)

    run_proc = subprocess.Popen(["./random_poly.out", str(radius), str(max_size)], stdout=subprocess.PIPE)
    run_out = run_proc.communicate()

    cordinates = run_out[0].decode("utf-8").split(" ")[1:-1]
    points = []
    for i in range(0, len(cordinates), 2):
        points.append((int(cordinates[i]) + radius, int(cordinates[i+1]) + radius))

    return points

def draw_polygon(points, radius):
    image = Image.new("RGB", (2*radius, 2*radius))

    draw = ImageDraw.Draw(image)
    draw.polygon((points), fill=(0,0,255,255))

    image.show()

def run():
    radius = 200
    max_size = 100

    points = get_random_polygon(radius, max_size)
    
    draw_polygon(points, radius)

if __name__ == '__main__':
    run()
