import os, glob, random, torch, cv2
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--phase_num", type=int, default=20)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--testset_name", type=str, default="Set11")
parser.add_argument("--result_dir", type=str, default="test_out")
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--num_feature", type=int, default=32)

args = parser.parse_args()
epoch = args.epoch
N_p = args.phase_num
B = args.block_size
nf = args.num_feature

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fixed seed for reproduction
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

N = B * B
cs_ratio_list = [0.01, 0.04, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

model = Net(N_p, B, nf)
model = torch.nn.DataParallel(model).to(device)
model_dir = "%s/layer_%d_block_%d_f_%d" % (args.model_dir, N_p, B, nf)
#model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (model_dir, epoch)))
model.load_state_dict(torch.load("%s/final_params.pkl" % (model_dir)))

result_dir = os.path.join(args.result_dir, str(args.phase_num), args.testset_name)
os.makedirs(result_dir, exist_ok=True)

# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + "/*")
test_image_num = len(test_image_paths)

def test(q_G, q_D):
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(test_image_num):
            image_path = test_image_paths[i]
            test_image = cv2.imread(image_path, 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0])
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            x_input = torch.from_numpy(img_pad)
            x_input = x_input.type(torch.FloatTensor).to(device)
            x_output = model(x_input, q_G, q_D)
            x_output = x_output.cpu().data.numpy().squeeze()
            x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0
            PSNR = psnr(x_output, img)
            SSIM = ssim(x_output, img, data_range=255)
            test_image_ycrcb[:,:,0] = x_output
            test_image = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.uint8)
            image_path = image_path.split("/")[-1]
            result_path = os.path.join(result_dir, image_path)
            cv2.imwrite("%s_ratio_%.2f_PSNR_%.2f_SSIM_%.4f.png" % (result_path, cs_ratio, PSNR, SSIM), test_image)
            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)
    return float(np.mean(PSNR_list)), float(np.mean(SSIM_list))

for cs_ratio in cs_ratio_list:
    q = np.ceil(cs_ratio * N)
    q_G = torch.tensor([[int(np.round(0.6 * q if q <= (N // 2) else q))]], device=device)
    q_D = q - q_G
    avg_psnr, avg_ssim = test(q_G, q_D)
    print("CS Ratio is %.2f, Q is %d, q_G is %d, q_D is %d, avg PSNR is %.2f, avg SSIM is %.4f." % (cs_ratio, q, int(q_G), int(q_D), avg_psnr, avg_ssim))