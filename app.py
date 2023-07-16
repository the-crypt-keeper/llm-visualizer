from transformers import AutoModelForCausalLM
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import numpy as np

st.set_page_config(page_title="LLM Visualizer", layout="wide")
st.markdown("""
    <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 0rem;
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
    </style>
    """, unsafe_allow_html=True)

model_col, tensor_col, colormap_col = st.columns(3)

with model_col:    
    models = ["gpt2"]
    model_name = st.selectbox("Select Model", models)

model = AutoModelForCausalLM.from_pretrained(model_name)
state_dict = model.state_dict()
tensors = list(state_dict.keys())
with tensor_col:
    tensor_name = st.selectbox("Select Tensor", tensors, index=tensors.index('lm_head.weight'))
with colormap_col:
    colormap = st.selectbox("Select Colormap", ['viridis', 'plasma', 'inferno', 'bwr', 'jet'], index=3)

clip_col, width_col, height_col = st.columns(3)

with clip_col:
    clip_min_col, clip_max_col = st.columns(2)
    clip_min = clip_min_col.number_input('Clip Min', value=-0.5, step=0.01, min_value=-1., max_value=1.)
    clip_max = clip_max_col.number_input('Clip Max', value=0.5, step=0.01, min_value=-1., max_value=1.)

    data = state_dict[tensor_name].numpy()
    clipped = np.clip(data, clip_min, clip_max) * 2 / (clip_max-clip_min)
    transpose = st.checkbox('Transpose', value=True)

trans = clipped.T if transpose else clipped
width = trans.shape[1] if len(trans.shape) > 1 else 1
height = trans.shape[0]

with width_col:
    if width > 1024:
        range_col, offset_col = st.columns(2)
        width_range = range_col.select_slider('Width Range', options=[512,1024,2048], value=1024)
        width_offset = offset_col.number_input('Width Offset', value=0, min_value=0, max_value=width//width_range-1)
    else:
        st.write('Width', width)
        width_range = width
        width_offset = 0

with height_col:
    if height > 1024:
        range_col, offset_col = st.columns(2)
        height_range = range_col.select_slider('Height Range', options=[512,1024,2048], value=1024)
        height_offset = offset_col.number_input('Height Offset', value=0, min_value=0, max_value=height//height_range-1)
    else:
        st.write('Height', height)
        height_range = height
        height_offset = 0

print(width_range, width_offset, height_range, height_offset)
if len(trans.shape) > 1:
    window = trans[height_offset*height_range:height_range*(height_offset+1), width_offset*width_range:width_range*(width_offset+1)]
else:
    window = trans[height_offset*height_range:height_range*(height_offset+1)]

def display_array(arr, colormap='bwr', width=None):
  """ display a 2D array in streamlit with a color map"""

  # color map
  cm = matplotlib.colormaps.get_cmap(colormap)
  rgb = cm(arr)

  # display in app
  st.image(rgb, width=width)

display_array(window, colormap=colormap, width = width_range)