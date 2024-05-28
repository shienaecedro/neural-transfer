import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import threading
import os
import webbrowser
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from imgurpython import ImgurClient


customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("theme/dark-blue.json")

app = customtkinter.CTk(fg_color="White")
app.geometry("702x560")
app.title("Photo Art Style Transfer")

clicked = False
styleinput = False
contentinput = False

neuralthread1 = lambda: threading.Thread(target=neural_style_transfer)

facebookthread = lambda: threading.Thread(target=share_on_facebook(imgur_link))
copythread = lambda: threading.Thread(target=copy_imgur_link(imgur_link))

def sharefbthread():
    fbthread = facebookthread()
    fbthread.start()

def sharecpthread():
    cpthread = copythread()
    cpthread.start()

def neuralthread():
    if not styleinput and not contentinput:
        error_label1 = customtkinter.CTkLabel(app, text="Input Content Image", text_color="red")
        error_label1.grid(row=4, column=2, padx=10, pady=5)
        app.after(2000, error_label1.destroy) 
        error_label2 = customtkinter.CTkLabel(app, text="Input Style Image", text_color="red")
        error_label2.grid(row=4, column=4, padx=10, pady=5)
        app.after(2000, error_label2.destroy)
    elif styleinput and not contentinput:
        error_label1 = customtkinter.CTkLabel(app, text="Input Content Image", text_color="red")
        error_label1.grid(row=4, column=2, padx=10, pady=5)
        app.after(2000, error_label1.destroy)
    elif contentinput and not styleinput:
        error_label2 = customtkinter.CTkLabel(app, text="Input Style Image", text_color="red")
        error_label2.grid(row=4, column=4, padx=10, pady=5)
        app.after(2000, error_label2.destroy)    
    else:
        global clicked
        clicked = False
        neural_thread = neuralthread1()  
        neural_thread.start()
        if neural_thread.is_alive():
            button3.configure(app, text="cancel_function", command=cancel_function)
            horizontal.grid_remove()
            horizontal_label.grid_remove()
            horizontal_label2.grid_remove()
            progress.grid(row=5, column=2, columnspan=2, pady=20)
            progress_percentage_label.grid(row=4, column=2, columnspan=2)

def slider_function(value):
    text_var.set(f'{int(horizontal.get())}')

def cancel_function():
    global clicked
    clicked = True
    button3.configure(text="Start Styling", command=neuralthread)
    progress.grid_remove()
    horizontal.grid(row=5, column=3)
    horizontal_label.grid(row=6, column=3)
    horizontal_label2.grid(row=4, column=3)

def create_top():
    global toplevel
    toplevel = customtkinter.CTkToplevel()
    toplevel.title("Photo Art Style Transfer")
    toplevel.geometry("600x415")
    toplevel.resizable(width=False, height=False)
    toplevel.configure(fg_color="White")
    top_widgets()

def top_widgets():

    output = Image.open(displayimg)
    displayoutput = customtkinter.CTkImage(output, size=(350, 400))
    image_label3 = customtkinter.CTkLabel(toplevel, image=displayoutput, text="",fg_color="White",width=355,height=405)
    image_label3.place(x = 5, y = 5)
    button_image_1 = tk.PhotoImage(
    file=("images/button_1.png"))

    share_label = customtkinter.CTkLabel(toplevel, text="Share Image", text_color="#1f538d", font=("Helvetica", 18))
    share_label.place(x=432.0, y=40.0)

    facebook_button = customtkinter.CTkButton(
        toplevel,text="", command=sharefbthread,image=button_image_1,border_width=0,fg_color="transparent",width=17
    )
    facebook_button.place(
    x=422.0,
    y=135.0,
    )
    button_image_2 = tk.PhotoImage(
        file=("images/button_2.png"))
    download_button = customtkinter.CTkButton(
        toplevel, text="", command=on_download_button_click, image=button_image_2, border_width=0, fg_color="transparent", width=17
    )
    download_button.place(
        x=422.0, 
        y=189.0
    )
    button_image_3 = tk.PhotoImage(
    file=("images/button_4.png"))
    copy_button = customtkinter.CTkButton(
        toplevel, text="", command=sharecpthread,image=button_image_3,border_width=0,fg_color="transparent",width=17
    )
    copy_button.place(
        x=422.0,
        y=81.0
    )
    button_image_4 = tk.PhotoImage(
    file=("images/button_3.png"))
    back_button = customtkinter.CTkButton(toplevel,text="",command=back_function,image=button_image_4,border_width=0,fg_color="transparent",width=17
    )
    back_button.place(
        x=422.0,
        y=243.0
    )

def content_function():
    global contentinput, contentpath
    contentpath = filedialog.askopenfilename(
        initialdir="/pictures", title="Select Image", 
        filetypes=(("jpg images", "*.jpg"), ("all files", "*.*"))
    )
    if contentpath:
        contentinput = True
        try:
            image = Image.open(contentpath)
            image_to_display1 = customtkinter.CTkImage(image, size=(300, 300))
            image_label1.configure(image=image_to_display1)
            image_label1.image = image_to_display1
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error opening image: {e}")

def style_function():
    global stylepath, styleinput
    stylepath = filedialog.askopenfilename(
        initialdir="/pictures", title="Select Image", 
        filetypes=(("jpg images", "*.jpg"), ("all files", "*.*"))
    )
    if stylepath:
        styleinput = True
        try:
            image = Image.open(stylepath)
            image_to_display2 = customtkinter.CTkImage(image, size=(300, 300))
            image_label2.configure(image=image_to_display2)
            image_label2.image = image_to_display2
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error opening image: {e}")

progress = customtkinter.CTkProgressBar(app)
progress_percentage_label = customtkinter.CTkLabel(app, text="0%", text_color="#1f538d")

def neural_style_transfer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    imsize = 512 if torch.cuda.is_available() else 128

    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])

    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader(stylepath)
    content_img = image_loader(contentpath)
    assert style_img.size() == content_img.size()

    plt.ion()

    class ContentLoss(nn.Module):
        def __init__(self, target):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.299, 0.224, 0.225])

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        normalization = Normalization(normalization_mean, normalization_std)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    input_img = content_img.clone()

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    numsteps = int(horizontal.get())

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=numsteps,
                           style_weight=1000000, content_weight=1):
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)

        input_img.requires_grad_(True)
        model.eval()
        model.requires_grad_(False)

        optimizer = get_input_optimizer(input_img)

        print('Optimizing..')
        progress["maximum"] = num_steps
        run = [0]
        while run[0] <= num_steps and not clicked:
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1

                global progress_value
                progress_value = run[0] / numsteps
                if progress_value > 1:
                    progress_value = 1
                progress.set(progress_value-0.03)
                progress_percentage_label.configure(text=f"{int(progress_value * 100)}%")

                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                progress.step()
                return style_score + content_score

            optimizer.step(closure)
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    plt.ioff()
    plt.show()
    global imgur_link
    out_t = (output.data.squeeze())
    output_img = transforms.ToPILImage()(out_t)

    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    counter = 0
    while os.path.exists(f'{outputs_dir}/output{counter}.png'):
        counter += 1

    global displayimg
    displayimg = f'{outputs_dir}/output{counter}.png'
    if not clicked:
        button3.configure(text="Start Styling", command=neuralthread)
        output_img.save(displayimg)
        progress.grid_remove()
        horizontal.grid(row=5, column=3)
        horizontal_label.grid(row=6, column=3)
        horizontal_label2.grid(row=4, column=3)
        app.withdraw()
        imgur_link = upload_image_to_imgur(displayimg)
        create_top()
        app.wait_window(toplevel)
        app.deiconify()

def back_function():
    toplevel.withdraw()
    app.deiconify()

def upload_image_to_imgur(displayimg):
    client_id = '9a3ec0ee4b97b64'
    client_secret = '4fba82309dd56809b0eaa54bd826c9f741a3033e'
    client = ImgurClient(client_id, client_secret)
    image = client.upload_from_path(displayimg, anon=True)
    return image['link']

def on_download_button_click():
    img = Image.open(displayimg)
    save_image(img)

def save_image(img):
    if img is None:
        messagebox.showerror("Error", "No image provided.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", ".png"),
                                                        ("All files", ".*")])
    if file_path:
        try:
            img.save(file_path)
            messagebox.showinfo("Success", f"Image saved: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save image: {e}")

def share_on_facebook(imgur_link):
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={imgur_link}"
    webbrowser.open(facebook_url, new=1)

def copy_imgur_link(imgur_link):
    try:
        app.clipboard_append(imgur_link)
        messagebox.showinfo("Success!", "Imgur link copied to clipboard.")
    except tk.TclError:
        messagebox.showerror("Error", "Failed to copy link.")


button1 = customtkinter.CTkButton(app, text="Select Content Image", command=content_function)
button1.grid(row=3, column=2, padx=5, pady=5)

placeholder_image1 = Image.open("images/imgholder.jpg")
placeholder_image_to_display1 = customtkinter.CTkImage(placeholder_image1, size=(300, 300))
image_label1 = customtkinter.CTkLabel(app, image=placeholder_image_to_display1, text="", fg_color="White", width=310, height=310)
image_label1.grid(row=2, column=2, padx=20, pady=10)

button2 = customtkinter.CTkButton(app, text="Select Style Image", command=style_function)
button2.grid(row=3, column=3, padx=5, pady=5) 

placeholder_image2 = Image.open("images/imgholder.jpg")
placeholder_image_to_display2 = customtkinter.CTkImage(placeholder_image2, size=(300, 300))
image_label2 = customtkinter.CTkLabel(app, image=placeholder_image_to_display2, text="", fg_color="White", width=310, height=310)
image_label2.grid(row=2, column=3, padx=20, pady=10)

horizontal = customtkinter.CTkSlider(app, from_=1, to=1000, command=slider_function)
horizontal.grid(row=5, column=2, columnspan=2, pady=10)
text_var = tk.StringVar(value="")

horizontal_label2 = customtkinter.CTkLabel(app, text="Image Quality Render", text_color="#1f538d")
horizontal_label2.grid(row=4, column=2, columnspan=2, pady=5)

horizontal_label = customtkinter.CTkLabel(app, textvariable=text_var, text_color="#1f538d")
horizontal_label.grid(row=6, column=2, columnspan=2, pady=5)

button3 = customtkinter.CTkButton(app, text="Start Styling", command=neuralthread)
button3.grid(row=7, column=2, columnspan=2, pady=10)

app.resizable(width=False, height=False)

app.mainloop()