from PIL import Image
import os
from IPython.display import display, Image as ipyImage
import ipywidgets as widgets
import os
from imagestack import ImageStack, ImageStackLazy


class UI:
    def __init__(self) -> None:
        pass
    
    def show(self, _type="pre"):
        if _type == "pre":
            self.pre()
        else:
            self.lazy()

    def pre(self):
        def update_filter_args(change):
            filter_name = filter_name_dropdown.value
            if filter_name == 'gaussian':
                filter_size_slider.layout.display = 'block'
                sigma_slider.layout.display = 'block'
            elif filter_name in ["mean", "median"]:
                filter_size_slider.layout.display = 'block'
                sigma_slider.layout.display = 'none'
            else:
                filter_size_slider.layout.display = 'none'
                sigma_slider.layout.display = 'none'
                filter_size_slider.value = 7
            
            filter_size_label.value = f'Filter Size: {filter_size_slider.value}'
            sigma_label.value = f'Sigma: {sigma_slider.value}'

        def update_image_dropdown(change):
            folder_path = folder_path_input.value
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                global img 
                with output:
                    output.clear_output(wait=True)
                    display(ipyImage(filename="assets/loading.gif"))
                    img = ImageStack(folder_path)
                    output.clear_output()
                image_dropdown.options = image_files
                image_dropdown.value = None
            else:
                image_dropdown.options = []
                image_dropdown.value = None

        def update_image(change):
            with output:
                output.clear_output(wait=True)  
                image_name = image_dropdown.value
                if image_name:
                    image_path = os.path.join(folder_path_input.value, image_name)

                    image_idx = img.names.index(image_name[:-4])

                    filter_args = {"filter_size": filter_size_slider.value}
                    if filter_name_dropdown.value == 'gaussian':
                        filter_args["sigma"] = sigma_slider.value
                    
                    Image.fromarray(img.apply_filter(None, filter_name_dropdown.value, filter_args, image_idx)).convert('L').save("cache.png")
                    display(ipyImage(filename="cache.png"))

        folder_path_input = widgets.Text(
            description='Folder Path:',
            placeholder='Enter folder path',
            layout=widgets.Layout(width='50%', height='40px')
        )

        image_dropdown = widgets.Dropdown(
            description='Select Image:',
            options=[],
            value=None,
            layout=widgets.Layout(width='50%', height='40px')
        )

        filter_name_dropdown = widgets.Dropdown(
            description='Filter Name:',
            options=["mean", "median", "prewitt_x", "prewitt_y", "sobel_x", "sobel_y", "laplacian", "gaussian", "log"],  # Replace with actual filter names
            value='sobel_x',
            layout=widgets.Layout(width='50%', height='40px')
        )

        filter_size_slider = widgets.IntSlider(
            value=7,
            min=1,
            max=7,
            step=2,
            description='Filter Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%', height='50px')
        )

        filter_size_label = widgets.Label(
            value=f'Filter Size: {filter_size_slider.value}',
            layout=widgets.Layout(width='50%', height='30px')
        )

        sigma_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=7,
            step=2,
            description='Sigma:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%', height='50px')
        )

        sigma_label = widgets.Label(
            value=f'Sigma: {sigma_slider.value}',
            layout=widgets.Layout(width='50%', height='30px')
        )

        output = widgets.Output()

        folder_path_input.observe(update_image_dropdown, names='value')
        image_dropdown.observe(update_image, names='value')
        filter_name_dropdown.observe(update_filter_args, names='value')
        filter_name_dropdown.observe(update_image, names='value')
        filter_size_slider.observe(update_image, names='value')
        sigma_slider.observe(update_image, names='value')

        sigma_slider.layout.display = 'none'
        sigma_label.layout.display = 'none'
        filter_size_slider.layout.display = 'none'
        filter_size_label.layout.display = 'none'


        title = widgets.HTML(value="<h1>Experiment 5 | UI with Multi-Core Precompute</h1>")
        subtitle = widgets.HTML(value="<h3>Irsh Vijay | 21EC39055 <br> Choose a folder, select an image, apply filters, and view the results. (Precompute takes approx 30-40s on 14 threads)</h3>")


        display(title, subtitle, folder_path_input, image_dropdown, filter_name_dropdown, filter_size_label, filter_size_slider, sigma_label, sigma_slider, output)
    
    def lazy(self):
        def update_filter_args(change):
            filter_name = filter_name_dropdown.value
            if filter_name == 'gaussian':
                filter_size_slider.layout.display = 'block'
                sigma_slider.layout.display = 'block'
            elif filter_name in ["mean", "median"]:
                filter_size_slider.layout.display = 'block'
                sigma_slider.layout.display = 'none'
            else:
                filter_size_slider.layout.display = 'none'
                sigma_slider.layout.display = 'none'
                filter_size_slider.value = 7
            
            filter_size_label.value = f'Filter Size: {filter_size_slider.value}'
            sigma_label.value = f'Sigma: {sigma_slider.value}'

        def update_image_dropdown(change):
            folder_path = folder_path_input.value
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                global img 
                img = ImageStackLazy(folder_path)
                image_dropdown.options = image_files
                image_dropdown.value = None
            else:
                image_dropdown.options = []
                image_dropdown.value = None

        def update_image(change):
            with output:
                output.clear_output(wait=True)  
                image_name = image_dropdown.value
                if image_name:
                    image_path = os.path.join(folder_path_input.value, image_name)

                    image_idx = img.names.index(image_name[:-4])

                    filter_args = {"filter_size": filter_size_slider.value}
                    if filter_name_dropdown.value == 'gaussian':
                        filter_args["sigma"] = sigma_slider.value
                    
                    Image.fromarray(img.apply_filter(None, filter_name_dropdown.value, filter_args, image_idx)).convert('L').save("cache.png")
                    display(ipyImage(filename="cache.png"))

        folder_path_input = widgets.Text(
            description='Folder Path:',
            placeholder='Enter folder path',
            layout=widgets.Layout(width='50%', height='40px')
        )

        image_dropdown = widgets.Dropdown(
            description='Select Image:',
            options=[],
            value=None,
            layout=widgets.Layout(width='50%', height='40px')
        )

        filter_name_dropdown = widgets.Dropdown(
            description='Filter Name:',
            options=["mean", "median", "prewitt_x", "prewitt_y", "sobel_x", "sobel_y", "laplacian", "gaussian", "log"],  # Replace with actual filter names
            value='sobel_x',
            layout=widgets.Layout(width='50%', height='40px')
        )

        filter_size_slider = widgets.IntSlider(
            value=7,
            min=1,
            max=7,
            step=2,
            description='Filter Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%', height='50px')
        )

        filter_size_label = widgets.Label(
            value=f'Filter Size: {filter_size_slider.value}',
            layout=widgets.Layout(width='50%', height='30px')
        )

        sigma_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=7,
            step=2,
            description='Sigma:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%', height='50px')
        )

        sigma_label = widgets.Label(
            value=f'Sigma: {sigma_slider.value}',
            layout=widgets.Layout(width='50%', height='30px')
        )

        output = widgets.Output()

        folder_path_input.observe(update_image_dropdown, names='value')
        image_dropdown.observe(update_image, names='value')
        filter_name_dropdown.observe(update_filter_args, names='value')
        filter_name_dropdown.observe(update_image, names='value')
        filter_size_slider.observe(update_image, names='value')
        sigma_slider.observe(update_image, names='value')

        sigma_slider.layout.display = 'none'
        sigma_label.layout.display = 'none'
        filter_size_slider.layout.display = 'none'
        filter_size_label.layout.display = 'none'


        title = widgets.HTML(value="<h1>Experiment 5 | UI with Multi-Core Precompute</h1>")
        subtitle = widgets.HTML(value="<h3>Irsh Vijay | 21EC39055 <br> Choose a folder, select an image, apply filters, and view the results. (Precompute takes approx 30-40s on 14 threads)</h3>")


        display(title, subtitle, folder_path_input, image_dropdown, filter_name_dropdown, filter_size_label, filter_size_slider, sigma_label, sigma_slider, output)