{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.layers import AveragePooling2D\n",
    "from tensorflow.keras.layers import Dropout\n",
    "from tensorflow.keras.layers import Flatten\n",
    "from tensorflow.keras.layers import Dense\n",
    "from tensorflow.keras.layers import Input\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
    "from tensorflow.keras.preprocessing.image import img_to_array\n",
    "from tensorflow.keras.preprocessing.image import load_img\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from sklearn.preprocessing import LabelBinarizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report\n",
    "from imutils import paths\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] loading images...\n"
     ]
    }
   ],
   "source": [
    "INIT_LR = 1e-4\n",
    "EPOCHS = 20\n",
    "BS = 32\n",
    "\n",
    "DIRECTORY = r\"C:\\Users\\Admin\\.conda\\envs\\mlprog1\\PranaviSquad\\major\\Face-Mask-Detection-master\\dataset\"\n",
    "CATEGORIES = [\"with_mask\", \"without_mask\"]\n",
    "\n",
    "# grab the list of images in our dataset directory, then initialize\n",
    "# the list of data (i.e., images) and class images\n",
    "print(\"[INFO] loading images...\")\n",
    "\n",
    "data = []\n",
    "labels = []\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\.conda\\envs\\mlprog1\\lib\\site-packages\\PIL\\Image.py:960: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "for category in CATEGORIES:\n",
    "    path = os.path.join(DIRECTORY, category)\n",
    "    for img in os.listdir(path):\n",
    "    \timg_path = os.path.join(path, img)\n",
    "    \timage = load_img(img_path, target_size=(224, 224))\n",
    "    \timage = img_to_array(image)\n",
    "    \timage = preprocess_input(image)\n",
    "\n",
    "    \tdata.append(image)\n",
    "    \tlabels.append(category)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "lb = LabelBinarizer()\n",
    "labels = lb.fit_transform(labels)\n",
    "labels = to_categorical(labels)\n",
    "\n",
    "data = np.array(data, dtype=\"float32\")\n",
    "labels = np.array(labels)\n",
    "\n",
    "(trainX, testX, trainY, testY) = train_test_split(data, labels,\n",
    "\ttest_size=0.20, stratify=labels, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.\n",
      "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5\n",
      "9412608/9406464 [==============================] - 1s 0us/step\n",
      "[INFO] compiling model...\n",
      "[INFO] training head...\n",
      "Epoch 1/20\n",
      "95/95 [==============================] - 218s 2s/step - loss: 0.5468 - accuracy: 0.7828 - val_loss: 0.1487 - val_accuracy: 0.9791\n",
      "Epoch 2/20\n",
      "95/95 [==============================] - 192s 2s/step - loss: 0.1632 - accuracy: 0.9675 - val_loss: 0.0718 - val_accuracy: 0.9896\n",
      "Epoch 3/20\n",
      "95/95 [==============================] - 175s 2s/step - loss: 0.1005 - accuracy: 0.9798 - val_loss: 0.0536 - val_accuracy: 0.9909\n",
      "Epoch 4/20\n",
      "95/95 [==============================] - 167s 2s/step - loss: 0.0743 - accuracy: 0.9796 - val_loss: 0.0456 - val_accuracy: 0.9922\n",
      "Epoch 5/20\n",
      "95/95 [==============================] - 166s 2s/step - loss: 0.0635 - accuracy: 0.9842 - val_loss: 0.0410 - val_accuracy: 0.9935\n",
      "Epoch 6/20\n",
      "95/95 [==============================] - 178s 2s/step - loss: 0.0576 - accuracy: 0.9848 - val_loss: 0.0379 - val_accuracy: 0.9922\n",
      "Epoch 7/20\n",
      "95/95 [==============================] - 169s 2s/step - loss: 0.0516 - accuracy: 0.9861 - val_loss: 0.0348 - val_accuracy: 0.9922\n",
      "Epoch 8/20\n",
      "95/95 [==============================] - 170s 2s/step - loss: 0.0497 - accuracy: 0.9846 - val_loss: 0.0349 - val_accuracy: 0.9909\n",
      "Epoch 9/20\n",
      "95/95 [==============================] - 170s 2s/step - loss: 0.0446 - accuracy: 0.9882 - val_loss: 0.0339 - val_accuracy: 0.9896\n",
      "Epoch 10/20\n",
      "95/95 [==============================] - 169s 2s/step - loss: 0.0350 - accuracy: 0.9902 - val_loss: 0.0317 - val_accuracy: 0.9922\n",
      "Epoch 11/20\n",
      "95/95 [==============================] - 165s 2s/step - loss: 0.0398 - accuracy: 0.9863 - val_loss: 0.0313 - val_accuracy: 0.9935\n",
      "Epoch 12/20\n",
      "95/95 [==============================] - 165s 2s/step - loss: 0.0386 - accuracy: 0.9893 - val_loss: 0.0306 - val_accuracy: 0.9922\n",
      "Epoch 13/20\n",
      "95/95 [==============================] - 165s 2s/step - loss: 0.0317 - accuracy: 0.9890 - val_loss: 0.0308 - val_accuracy: 0.9922\n",
      "Epoch 14/20\n",
      "95/95 [==============================] - 165s 2s/step - loss: 0.0463 - accuracy: 0.9826 - val_loss: 0.0296 - val_accuracy: 0.9935\n",
      "Epoch 15/20\n",
      "95/95 [==============================] - 165s 2s/step - loss: 0.0345 - accuracy: 0.9921 - val_loss: 0.0302 - val_accuracy: 0.9909\n",
      "Epoch 16/20\n",
      "95/95 [==============================] - 165s 2s/step - loss: 0.0384 - accuracy: 0.9899 - val_loss: 0.0286 - val_accuracy: 0.9909\n",
      "Epoch 17/20\n",
      "95/95 [==============================] - 166s 2s/step - loss: 0.0275 - accuracy: 0.9925 - val_loss: 0.0277 - val_accuracy: 0.9935\n",
      "Epoch 18/20\n",
      "95/95 [==============================] - 1135s 12s/step - loss: 0.0259 - accuracy: 0.9944 - val_loss: 0.0278 - val_accuracy: 0.9909\n",
      "Epoch 19/20\n",
      "95/95 [==============================] - 167s 2s/step - loss: 0.0332 - accuracy: 0.9881 - val_loss: 0.0295 - val_accuracy: 0.9922\n",
      "Epoch 20/20\n",
      "95/95 [==============================] - 168s 2s/step - loss: 0.0207 - accuracy: 0.9965 - val_loss: 0.0261 - val_accuracy: 0.9935\n",
      "[INFO] evaluating network...\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "   with_mask       0.99      0.99      0.99       383\n",
      "without_mask       0.99      0.99      0.99       384\n",
      "\n",
      "    accuracy                           0.99       767\n",
      "   macro avg       0.99      0.99      0.99       767\n",
      "weighted avg       0.99      0.99      0.99       767\n",
      "\n",
      "[INFO] saving mask detector model...\n"
     ]
    },
    {
     "data": {
      "image/png": "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k7/97W/s27fP/+RbQ4I5PldddRWLFy/moosuYvHixfTu3Zv//e9/5OXlceWVVwL2lcTPf/5z5s2bh9Pp5KqrrmrW8RXH6dASDdHuGiporq8wdt68eSopKUmFh4erCy64QP39738PWLehgubjCwENw1ArVqxocHv1bX/cuHHquuuu84/n5eWpyy67TIWFhanExER11113qalTp6oLL7yw0f0F6u3uv/9+pZRSK1euVP3791cOh0N169ZN3XHHHaqyslIppVR2dra65JJLVPfu3ZXT6VRdu3ZVM2fOVEePHlVKKfXCCy+os88+W3Xp0kWFhYWpgQMHqqeeeqrReP71r3+pU089VblcLnXGGWeod955J+D4VBc0b968OWC93r17q/nz5/vH9+7dq8aPH69cLpfq3r27evjhh9Xo0aMbLGjevn17nQL/2v785z+r8PBw/749++yz6vTTT1dOp1PFxcWpiRMn+gujLctSDz/8sOrbt69yOBwqKSlJTZ061f9ZRUVF6pprrlFdunRRiYmJav78+fUWNNcXa1PHRyn74YVp06ap+Ph45XK5VL9+/QLmK6XU4cOHlcPhUD//+c/r3V8RPE0paXlNdH4+n4/+/ftz0UUXsWTJko4OR3QyX375JQMHDuSTTz5h6NChHR3OD5rcPhKd0qZNm8jNzeXMM8+kuLiYhx56iG+//Zbrr7++o0MTnUhFRQUHDhxg7ty5jB49WhJCG5CkIDoln8/HggULyMrKwuFwMGjQIDZu3Fjv/XFx8vrHP/7B9OnTGThwIP/+9787OpwTgtw+EkII4SePpAohhPCTpCCEEMLvB1+mcPxLM8FKSEggLy+vjaNpO509Puj8MUp8rSPxtU5njq+xNlHkSkEIIYSfJAUhhBB+khSEEEL4SVIQQgjhF5KC5scff5xt27YRExNTbxUFSilWrFjB9u3bcblczJo1K6DWSiGEEKERkiuFMWPGcMcddzQ4f/v27eTk5PDoo4/y85//nKeeeioUYQkhhDhOSJLCgAEDiIyMbHD+J598wqhRo9A0jb59+1JSUhLQaLgQQojQ6BTvKRQUFJCQkOAfj4+Pp6CggNjY2DrLrl+/nvXr1wOwaNGigPWawzTNFq8bCi2Jz7KsBjufzxcwrqpa89J13d8ZhhEwfnzXFjGGUqjja+z41/d3yMvLQylV5zg39Hdob0qp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vHkza9asoU+fPowaNYqMjAzcbnd7ximEECIEmtXymq7rDB06lKFDh7J//34effRRHn/8cZ566inOOeccrrjiCuLi4torViGEEO2sWUmhtLSULVu2sHnzZvbt20dGRgYzZswgISGB1157jfvuu48//vGP7RWrEEKIdhZ0UliyZAk7d+7ktNNOY/z48QwbNgyHw+Gff+2113L99de3R4xCCCFCJOik0KdPH2bMmEGXLl3qna/rOk8++WSbBSaEECL0gn4k9fTTT8fr9QZMy8vLC3hM1eVytVlgQgghQi/opPDYY4/h8/kCpnm9Xv785z+3eVBCCCE6RtBJIS8vj+Tk5IBpKSkpHD58uM2DEkII0TGCTgpxcXH873//C5j2v//9j9jY2DYPSgghRMcIuqB50qRJPPjgg1x00UUkJydz6NAhXn31VS699NL2jE8IIUQIBZ0UMjMziYiIYMOGDeTn5xMfH8+1117L8OHD2zM+IYQQIdSsl9fOPvtszj777PaKRQghRAdrVlI4evQoWVlZFBcXo5TyTz/vvPPaPDAhhBChF3RS+Pjjj3nsscfo2rUr+/fvJy0tjf3799O/f39JCkIIcYIIOik899xzzJo1i7PPPpsbbriBxYsXs3HjRvbv39+e8QkhhAihoJNCXl5enfKE0aNH8/Of/5xrr722yfV37NjBihUrsCyLcePGcfHFF9dZ5osvvmDlypX4fD6ioqK45557gg1PCCFEGwg6KURHR3P06FG6dOlCYmIiu3fvJioqCsuymlzXsiyWLVvGvHnziI+PZ+7cuaSnp5OamupfpqSkhKeeeoo777yThIQECgsLW7ZHQgghWizopDBu3Di+/vprhg8fzqRJk7jnnnvQNI0LL7ywyXWzsrJISUnxvxE9YsQItm7dGpAU3nvvPTIyMkhISAAgJiamufsihBCilTRV+zGiRliWha7XvACdl5dHeXl5wIm9IVu2bGHHjh3cfPPNAGzatIk9e/YwY8YM/zIrV67E6/Xy/fffU1ZWxsSJExk9enSdz1q/fj3r168HYNGiRXg8nmDCr8M0zToV/HUmnT0+6PwxSnytI/G1TmeOz+l0NjgvqCsFy7KYNm0aK1eu9LehUP2LPhj15R1N0wLGfT4fe/fu5a677sLj8TBv3jz69OlDt27dApbLzMwkMzPTP56Xlxd0HLUlJCS0eN1Q6OzxQeePUeJrHYmvdTpzfMefV2sLqu4jXdfp1q0bxcXFLQogPj6e/Px8/3h+fn6dOpPi4+MZPHgwbreb6OhoTjvtNPbt29ei7QkhhGiZoCvEGzlyJA888ADvvPMOn332GZ9//rm/a0rv3r05ePAgubm5eL1ePvjgA9LT0wOWSU9P5+uvv8bn81FRUUFWVhbdu3dv/h4JIYRosaALmt966y0A/vWvfwVM1zStyTYVDMNg+vTpLFy4EMuyGDt2LGlpaf7PnDBhAqmpqZxxxhnceuut6LrOeeedR48ePZq7P0IIIVoh6ILmzio7O7tF63Xm+33Q+eODzh+jxNc6El/rdOb4Wl2mIIQQ4uQQ9O2jX/ziFw3O++tf/9omwQghhOhYQSeFX/3qVwHjR44c4Y033uCcc85p86CEEEJ0jKCTwoABA+pMGzhwIAsXLmTixIltGpQQQoiO0aoyBdM0yc3NbatYhBBCdLBmVZ1dW0VFBdu3b+fMM89s86CEEEJ0jKCTQu03kgFcLhcXXngho0aNavOghBBCdIygk8KsWbPaMw4hhBCdQNBlCi+99BJZWVkB07Kysnj55ZfbPCghhBAdI+ik8MYbb9SpJjs1NZU33nijzYMSQgjRMYJOCl6vF9MMvNtkmmaL2zMQQgjR+QSdFHr16sV//vOfgGlvvfUWvXr1avOghBBCdIygC5qvu+46FixYwKZNm0hOTubQoUMcPXqUu+66qz3jE0IIEUJBJ4W0tDQeeeQRPv30U/Lz88nIyGDo0KG43e72jE8IIUQIBZ0UCgoKcDqdAXUdHTt2jIKCAuLi4tolOCGEEKEVdJnCgw8+SEFBQcC0goIC/vjHP7Z5UEIIITpG0EkhOzu7TktoPXr04MCBA20elBBCiI4RdFKIjo4mJycnYFpOTg5RUVFtHpQQQoiOEXSZwtixY1myZAk//elPSU5OJicnh+eee47zzjuvPeMTQggRQkEnhYsvvhjTNFm1ahX5+fnEx8dz3nnnMXny5PaMTwghRAgFnRR0Xeeiiy7ioosu8k+zLIvt27czZMiQdglOCCFEaAWdFGrbt28f7777Lu+99x6WZfHUU0+1dVxCCCE6QNBJoaioiM2bN/Puu++yb98+NE3jhhtukDIFIYQ4gTSZFLZs2cI777zDzp076d69OyNHjuS2227jzjvvZPjw4TgcjlDEKYQQIgSaTAoPPfQQkZGRzJkzh7POOisUMQkhhOggTSaFX/ziF7z77rv86U9/onfv3owcOZIRI0agaVoo4hNCCBFCTSaFMWPGMGbMGA4fPsy7777Lm2++yTPPPAPA9u3bGTVqFLoe9DtwQgghOrGgC5oTExOZOnUqU6dO5euvv+bdd9/l6aef5h//+AdLly5tzxiFEEKESJNJ4b///S8DBgwIaHWtf//+9O/fn+nTp7N169Z2DVAIIUToNJkUXn31VR555BH69evHkCFDGDJkiL+qbIfDwYgRI9o9SCGEEKHRZFK48847qaio4LPPPmP79heBWrMAACAASURBVO28+OKLhIeHc+aZZzJkyBD69u0rZQpCCHGCCKpMweVykZ6eTnp6OgDfffcd27dv5x//+AfZ2dkMHDiQSZMm0adPn3YNVgghRPtqUTUXPXr0oEePHkyZMoXS0lJ27txJWVlZo+vs2LGDFStWYFkW48aN4+KLL653uaysLO68807mzJnD8OHDWxKeEEKIFgo6KXz++eckJSWRlJTEkSNHWL16NYZhcNVVV3H22Wc3uq5lWSxbtox58+YRHx/P3LlzSU9PJzU1tc5yq1ev5owzzmjZ3gghhGiVoAsDli1b5i87eOaZZ/D5fABBPY6alZVFSkoKycnJmKbJiBEj6n1qae3atWRkZBAdHR1sWC2mKivbfRtCCPFDE/SVQkFBAQkJCfh8Pnbu3Mnjjz+OaZrcdNNNQa0bHx/vH4+Pj2fPnj11lvn444+ZP38+f/3rXxv8rPXr17N+/XoAFi1aREJCQrC74Ff+4Tscfmwh8Y/9HSM+sdnrh4Jpmi3at1Dq7DFKfK0j8bVOZ4+vIUEnhbCwMI4ePcr+/ftJTU3F7Xbj9Xrxer1NrquUqjPt+GoyVq5cydVXX93kk0yZmZlkZmb6x/Py8oLcg1rxRESjykrI37gWfczEZq8fCgkJCS3at1Dq7DFKfK0j8bVOZ46vW7duDc4LOimcf/75zJ07F6/Xy/XXXw/A119/Tffu3ZtcNz4+nvz8fP94fn4+sbGxAct88803PPLII4BdTff27dvRdb19KuHrmobRNQ3f9o+gkyYFIYToCM1qjvOss85C13VSUlIAiIuL4+abb25y3d69e3Pw4EFyc3OJi4vjgw8+YPbs2QHL/OUvfwkYHjp0aLvVyqppGq6zzqX0tedRpSVo4RHtsh0hhPihadYjqbUvOT7//HN0XWfAgAFNrmcYBtOnT2fhwoVYlsXYsWNJS0vjrbfeAmDChAnNDLv1XGedS+nLf0d9sQ1t2Lkh374QQnRGQSeF+fPnc9VVV9G/f39eeuklXn/9dXRd5yc/+QmXXnppk+tXV5FRW0PJ4JZbbgk2rBZz9BsEUTGw4yOQpCCEEEAzHkndv38/ffv2BeDtt99m/vz5LFy4kHXr1rVbcO1JMwy004ehPvsE5ZXHU4UQApqRFKqfIMrJyQEgNTWVhIQESkpK2ieyENDOyICyUtj9eUeHIoQQnULQt4/69evH8uXLOXLkCMOGDQPsBBEVFdVuwbW7AWeA04Xa8RHagDM7OhohhOhwQV8p3HLLLYSHh9OzZ0+uuOIKALKzs5k48Yf7SKfmdMGAM1E7Pq73XQohhDjZBH2lEBUVxc9+9rOAaccXHP8QaWdkoHZsge++gZ6ndnQ4QgjRoYJOCl6vlzVr1rBp0yaOHDlCbGwso0aN4tJLLw1ole2HRjt9GErT7VtIkhSEECe5oM/mzz77LN988w033ngjiYmJHD58mBdeeIHS0lL/G84/RFpUNPQ5DbV9C0y5uqPDEUKIDhV0mcKWLVv47W9/y+DBg+nWrRuDBw/m1ltv5cMPP2zP+EJCG5wBB/ahDud0dChCCNGhmv1I6olIOyMDALXzow6ORAghOlbQt4/OPvtsHnjgAaZOneqv/e+FF15osoGdUFNKUV5ejmVZdWpire3QoUNUVFTYI5ExWNN/DaYDvbQ0RJE2LiC+DqCUQtd13G53o8dRCHFiCTopXHPNNbzwwgssW7aMI0eOEBcXx4gRI4KqOjuUysvLcTgcTRZ+m6aJYRj+cdX/x1B4BFwutFrTO8rx8XUEr9dLeXk5YWFhHRqHECJ0gk4Kpmly5ZVXcuWVV/qneTwepk2bxjXXXNMuwbWEZVktexoqPAIKC6CsBCLbv+W3HwLTNDv0akUIEXpBlynUpzPeVmhxTE4XGCaU/nCr7WgPnfFvLIRoP61KCicSTdPsq4WyUpRldXQ4QgjRIZq8z/L55w1XFtfZyhNaLTwSiguhvMxOEEIIcZJpMin89a9/bXT+D7Fh6ga53aDrUHqsxUmhsLCQF198sdkv9E2bNo0///nPxMTENGu9//f//h+ZmZlceOGFzVpPCCHq02RSqN1M5olO03RUWASUlaCUatH99KKiIp555pk6ScHn8zX6NNGqVauavS0hhGhrP9xKi4Jg/fNJ1P699c/TtPpfyPP5oNJjFzzrdYtctLRT0H96Y4PbvO+++9i3bx/jx4/H4XAQHh5OcnIyX3zxBe+88w7Tp08nOzubiooKZsyY4X9yKyMjg7Vr11JSUsI111xDRkYGW7duJSUlheXLlwf1WOjmzZu599578fl8DB48mPvvvx+Xy8V9993HW2+9hWmajBo1it///ve8+uqrPPTQQ+i6TnR0NGvWrGny84UQJ74TOim0SHUisHz1JoWm3HHHHezatYt169bxwQcfcO2117JhwwZ69OgBwJIlS4iNjaWsrIxJkyYxceJE4uLiAj5j7969LF26lMWLF3PTTTfxxhtvcNlllzW63fLycubMmcNzzz1H7969mT17Ns888wxTp05l7dq1bNq0CU3TKCwsBODhhx9m9erVdO3a1T9NCCFO6KTQ2C960zQbLChXhw6A14vWvWerYzjjjDP8CQFg+fLlrF27FrDbo9i7d2+dpJCWlsagQYPwer2cfvrp7N+/v8ntfPPNN/To0YPevXsDcPnll/P0009zww034HK5uPXWWxk3bhyZmZkApKenM2fOHCZPnswFF1zQ6v0UQpwY5JHU+oRFQqUH5fG0+qPCw8P9wx988AGbN2/m1VdfZf369QwaNKjel8NcLpd/2DAMfD5fk9tpqG4q0zR5/fXXmThxIm+++SZXX23XBPvAAw/w29/+luzsbCZMmEBBQUFzd00IcQI6oa8UWiw8HAqAsmPgjGty8doiIiI4duxYvfOKi4uJiYkhLCyMrKwstm3b1gbB2k499VT279/P3r17OeWUU3jhhRcYPnw4JSUllJWVMW7cOIYMGcLIkSMB+PbbbxkyZAhDhgxh3bp1ZGdn17liEUKcfCQp1EMzHSin2367OaZ5J8q4uDiGDRvGeeedh9vtDnhkd8yYMaxatYrMzEx69erVpi3Xud1u/vSnP3HTTTf5C5qnTZvG0aNHmT59OhUVFSilmD9/PgALFixg7969KKUYOXIkAwcObLNYhBA/XJr6gdeJnZ2dHTBeWloacMumIY2VKQCoowVwtABSf4TWAS3LNRVfqDR2PKtry+2sJL7WkfhapzPH161btwbnSZlCQ8IjAGVXkCeEECcJuX3UEIcTTId9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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "aug = ImageDataGenerator(\n",
    "\trotation_range=20,\n",
    "\tzoom_range=0.15,\n",
    "\twidth_shift_range=0.2,\n",
    "\theight_shift_range=0.2,\n",
    "\tshear_range=0.15,\n",
    "\thorizontal_flip=True,\n",
    "\tfill_mode=\"nearest\")\n",
    "\n",
    "# load the MobileNetV2 network, ensuring the head FC layer sets are\n",
    "# left off\n",
    "baseModel = MobileNetV2(weights=\"imagenet\", include_top=False,\n",
    "\tinput_tensor=Input(shape=(224, 224, 3)))\n",
    "\n",
    "# construct the head of the model that will be placed on top of the\n",
    "# the base model\n",
    "headModel = baseModel.output\n",
    "headModel = AveragePooling2D(pool_size=(7, 7))(headModel)\n",
    "headModel = Flatten(name=\"flatten\")(headModel)\n",
    "headModel = Dense(128, activation=\"relu\")(headModel)\n",
    "headModel = Dropout(0.5)(headModel)\n",
    "headModel = Dense(2, activation=\"softmax\")(headModel)\n",
    "\n",
    "# place the head FC model on top of the base model (this will become\n",
    "# the actual model we will train)\n",
    "model = Model(inputs=baseModel.input, outputs=headModel)\n",
    "\n",
    "# loop over all layers in the base model and freeze them so they will\n",
    "# *not* be updated during the first training process\n",
    "for layer in baseModel.layers:\n",
    "\tlayer.trainable = False\n",
    "\n",
    "# compile our model\n",
    "print(\"[INFO] compiling model...\")\n",
    "opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)\n",
    "model.compile(loss=\"binary_crossentropy\", optimizer=opt,\n",
    "\tmetrics=[\"accuracy\"])\n",
    "\n",
    "# train the head of the network\n",
    "print(\"[INFO] training head...\")\n",
    "H = model.fit(\n",
    "\taug.flow(trainX, trainY, batch_size=BS),\n",
    "\tsteps_per_epoch=len(trainX) // BS,\n",
    "\tvalidation_data=(testX, testY),\n",
    "\tvalidation_steps=len(testX) // BS,\n",
    "\tepochs=EPOCHS)\n",
    "\n",
    "# make predictions on the testing set\n",
    "print(\"[INFO] evaluating network...\")\n",
    "predIdxs = model.predict(testX, batch_size=BS)\n",
    "\n",
    "# for each image in the testing set we need to find the index of the\n",
    "# label with corresponding largest predicted probability\n",
    "predIdxs = np.argmax(predIdxs, axis=1)\n",
    "\n",
    "# show a nicely formatted classification report\n",
    "print(classification_report(testY.argmax(axis=1), predIdxs,\n",
    "\ttarget_names=lb.classes_))\n",
    "\n",
    "# serialize the model to disk\n",
    "print(\"[INFO] saving mask detector model...\")\n",
    "model.save(\"mask_detector.model\", save_format=\"h5\")\n",
    "\n",
    "# plot the training loss and accuracy\n",
    "N = EPOCHS\n",
    "plt.style.use(\"ggplot\")\n",
    "plt.figure()\n",
    "plt.plot(np.arange(0, N), H.history[\"loss\"], label=\"train_loss\")\n",
    "plt.plot(np.arange(0, N), H.history[\"val_loss\"], label=\"val_loss\")\n",
    "plt.plot(np.arange(0, N), H.history[\"accuracy\"], label=\"train_acc\")\n",
    "plt.plot(np.arange(0, N), H.history[\"val_accuracy\"], label=\"val_acc\")\n",
    "plt.title(\"Training Loss and Accuracy\")\n",
    "plt.xlabel(\"Epoch #\")\n",
    "plt.ylabel(\"Loss/Accuracy\")\n",
    "plt.legend(loc=\"lower left\")\n",
    "plt.savefig(\"plot.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
