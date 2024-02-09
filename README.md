# EVMF
This is an article about Video captioning. The article is called Multimodal Fusion Transformer and a novel modality for video captioning
## Pre
You need to download the features of the dataset.

Download features [I3D (17GB)](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/mdvc/sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5), [VGGish (1GB)](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/mdvc/sub_activitynet_v1-3.vggish.hdf5) and put in `./data/` folder (speech segments are already there). You may use `curl -O <link>` to download the features.

I have put the text for the Enhanced Visual mode and the Speech mode into the data folder.

Thank you very much for providing the open source code to help us extract the I3D features and VGGISH features：[video_features on GitHub](https://github.com/v-iashin/video_features/tree/6190f3d7db6612771b910cf64e274aedba8f1e1b) (make sure to check out to `6190f3d7db6612771b910cf64e274aedba8f1e1b` commit).

If you want to extract the original video for training and feature extraction, you can visit: [ActivityNet/ActivityNet@7185a39](https://github.com/activitynet/ActivityNet/tree/7185a3919e7baef8ad7311399f96e16d51546c53)

## Train 

You can train with the following code in the IDE or CMD

```bash
python main.py
```

## Evaluation Scrips and Results

You can use the official script to evaluate:[pycocoevalcap](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c)


## More information and details will be released soon.
