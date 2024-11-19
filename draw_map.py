import tkinter as tk
import tkinter.messagebox as messagebox
import argparse

class MapDrawer:
    def __init__(self, map_file):
        self.root = tk.Tk()
        self.root.title("地图路径绘制")

        self.map_image = tk.PhotoImage(file=map_file)
        w, h = self.map_image.width(), self.map_image.height()

        self.canvas = tk.Canvas(self.root, width=w, height=h)
        self.canvas.pack(expand=True, fill=tk.BOTH)


        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_image)

        self.canvas.bind("<Button-1>", self.add_yellow)
        self.canvas.bind("<Button-3>", self.add_green)
        self.root.bind("<BackSpace>", self.delete_last_point)

        self.points = []
        self.arrows = []
        self.scale = 1.0

        button_frame = tk.Frame(self.root)
        button_frame.pack()

        tk.Button(button_frame, text="save", command=self.generate_script).pack(side=tk.LEFT)
        tk.Button(button_frame, text="clear", command=self.clear_points).pack(side=tk.LEFT)

        self.root.mainloop()

    def add_point(self, x, y, color):
        point = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color)
        self.points.append((point, color, x, y))

        if len(self.points) > 1:
            prev_point = self.points[-2]
            arrow = self.canvas.create_line(prev_point[2], prev_point[3], x, y, arrow=tk.LAST)
            self.arrows.append(arrow)

    def add_yellow(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.add_point(x, y, "yellow")

    def add_green(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.add_point(x, y, "green")

    def delete_last_point(self, event):
        if self.points:
            point = self.points.pop()
            self.canvas.delete(point[0])

            if self.arrows:
                arrow = self.arrows.pop()
                self.canvas.delete(arrow)

    def zoom(self, event):
        if event.delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        print(self.scale)

        self.canvas.scale(tk.ALL, 0, 0, self.scale, self.scale)
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

    def clear_points(self):
        for point, _, _, _ in self.points:
            self.canvas.delete(point)
        for arrow in self.arrows:
            self.canvas.delete(arrow)
        self.points.clear()
        self.arrows.clear()

    def generate_script(self):
        script = []
        for _, color, x, y in self.points:
            if color == "yellow":
                script.append(f"('goto', {x}, {y}),")
            elif color == "green":
                script.append(f"('goto', {x}, {y}),\n('pick_loop', 'all'),")
            else:
                raise ValueError(f"Unknown color: {color}")

        script_str = "["+"\n".join(script)+"]"
        with open("path_script.txt", "w") as file:
            file.write(script_str)

        messagebox.showinfo("生成脚本", "path save to path_script.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='agent argument')
    parser.add_argument('map_id', type=str, default='0209')
    args = parser.parse_args()

    map_file = f"maps/{args.map_id}.png"  # 替换为你的地图图片文件路径
    MapDrawer(map_file)
