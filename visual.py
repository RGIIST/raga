import io, os
import base64
import pickle
import cv2
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default = 'light_met_crop_lmff1_dash.pickle', help = 'path/to the/pickle_file')
args = parser.parse_args()


color_list = ["#E52B50", "#9F2B68", "#3B7A57", "#3DDC84", "#FFBF00", "#FFBF00", "#915C83", "#008000"]

with open(args.path ,'rb') as f:
    visual = pickle.load(f)


width, height = 1400, 640
crop_name_a, images_a, labels_a, embed1_a = visual['crop_name'], visual['image'], visual['label'], np.array(visual['embed'])

classes = list(set(labels_a))

app = Dash(__name__)

color_a=[]
for l in labels_a:
    color_a.append(color_list[classes.index(l)])


crop_name, images, labels, embed1, colors = crop_name_a, images_a, labels_a, embed1_a, color_a

fig = go.Figure(data=[go.Scatter(
    x=embed1[:,0],
    y=embed1[:,1],
    mode='markers',
    marker=dict(
        size=2,
        color=colors,
    )
)],
    layout= {'autosize':False,'width':width,'height':height})

@app.callback(
    Output('graph-5', 'figure'),
    [Input('weather', 'value')])
def data(cls):
    global crop_name, images, labels, embed1, colors
    crop_name, images, labels, embed1, colors = [], [], [], [], []

    for l in range(len(labels_a)):
        if labels_a[l] in cls:
            colors.append(color_a[l])
            images.append(images_a[l])
            embed1.append(embed1_a[l].tolist())
            labels.append(labels_a[l])
            crop_name.append(crop_name_a[l])

    embed1 = np.array(embed1)
    return {
        'data':[go.Scatter(
            x=embed1[:,0],
            y=embed1[:,1],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
            )
        )],
        'layout': {'autosize':False,'width':width,'height':height}
    }


def np_image_to_base64(im_matrix):
    im_matrix =cv2.cvtColor(im_matrix, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

app.layout = html.Div(   
    className="container",
    children=[
        dcc.Dropdown(classes,
         classes, id='weather', multi= True,style={'width': '75%', 'display': 'inline-block'}),
        dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),  
    ],
    
)

@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = np.array(images[num],dtype=np.uint8)
    im_url = np_image_to_base64(im_matrix)
    label=crop_name[num]# +', ' + FP[num]
    
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "250px","height":"150px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P(label),
        ])
    ]
  
    return True, bbox, children

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True)
