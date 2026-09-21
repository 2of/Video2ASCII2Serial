# Video to ASCII to HTTPS to Serial



Goal is to render a video feed in a useful way by converting it to ASCII, serving it over HTTPS, and sending it over a serial connection. Will require changes buffer 

## Early Comparisons

Below are some early comparison screenshots showing the progression of the ASCII rendering:

#### straight up luminance 

![Comparison 1](./git/ss1.png)


![Comparison 2](./git/ss2.png)


#### composite of depth map + edge map + luminance map

![Comparison 3](./git/ss3.png)



![Comparison 4](./git/ss4.png)

## Usage

Install the required Python dependencies with:

```bash
pip3 install -r requirements.txt