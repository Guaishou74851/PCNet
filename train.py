import torch, os, glob, cv2, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser(description="PCT")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=1000)
parser.add_argument("--phase_num", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--testset_name", type=str, default="Set11")
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--num_feature", type=int, default=32)

args = parser.parse_args()

start_epoch, end_epoch = args.start_epoch, args.end_epoch
learning_rate = args.learning_rate
N_p = args.phase_num
B = args.block_size
nf = args.num_feature

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

gpu_num = torch.cuda.device_count()
print("device =", device)
print("gpu_num =", gpu_num)

# fixed seed for reproduction
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size = 32
patch_size = 128
iter_num = 1000
N = B * B
cs_ratio_list = [0.01, 0.04, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

# training set info
print("reading files...")
start_time = time()
training_image_paths = glob.glob(os.path.join(args.data_dir, "pristine_images") + "/*")
print("training_image_num", len(training_image_paths))

model = Net(N_p, B, nf)
model = torch.nn.DataParallel(model).to(device)

class MyDataset(Dataset):
    def __getitem__(self, index):
        path = random.choice(training_image_paths)
        x = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2YCrCb)
        x = torch.from_numpy(x[:, :, 0]) / 255.0
        h, w = x.shape
        max_h, max_w = h - patch_size, w - patch_size
        start_h = random.randint(0, max_h)
        start_w = random.randint(0, max_w)
        return x[start_h:start_h+patch_size, start_w:start_w+patch_size]

    def __len__(self):
        return iter_num * batch_size

def lr_func(z):
    warm_up_epoch = 10
    cos_epoch = 640
    ft1_epoch = 200
    ft2_epoch = 100
    ft3_epoch = 50
    T_max = 2e-4
    T_min = 1e-4
    ft1_lr = 1e-4
    ft2_lr = 1e-5
    ft3_lr = 1e-6
    t = warm_up_epoch
    if z <= t:
        return ((z + 1) / (warm_up_epoch + 1)) * T_max
    t += cos_epoch
    if z <= t:
        return np.cos((np.pi / 2) * (z - warm_up_epoch) / cos_epoch) * (T_max - T_min) + T_min
    t += ft1_epoch
    if z <= t:
        return ft1_lr
    t += ft2_epoch
    if z <= t:
        return ft2_lr
    t += ft3_epoch
    if z <= t:
        return ft3_lr

dataloader = DataLoader(dataset=MyDataset(), batch_size=batch_size, num_workers=8, pin_memory=True)
optimizer = torch.optim.AdamW([{"params":model.parameters(),"initial_lr":learning_rate}], lr=learning_rate, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, last_epoch=start_epoch-1)

model_dir = "./%s/layer_%d_block_%d_f_%d" % (args.model_dir, N_p, B, nf)
log_path = "./%s/layer_%d_block_%d_f_%d.txt" % (args.log_dir, N_p, B, nf)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + "/*")

def test(cs_ratio):
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(len(test_image_paths)):
            test_image = cv2.imread(test_image_paths[i], 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0], block_size=B)
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            x_input = torch.from_numpy(img_pad).to(device).float()
            q = (torch.tensor([[cs_ratio * N]], device=device)).ceil()
            q_G = (0.6 * q).round()
            q_DCT = q - q_G
            x_output = model(x_input, q_G, q_DCT)
            x_output = x_output.cpu().data.numpy().squeeze()
            x_output = np.clip(x_output[:old_h, :old_w], 0, 1) * 255.0
            PSNR = psnr(x_output, img)
            SSIM = ssim(x_output, img, data_range=255)
            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)
    return np.mean(PSNR_list), np.mean(SSIM_list)

if start_epoch > 0:
    model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (model_dir, start_epoch)))

print("start training...")
best_psnr = 0.0
best_epoch = -1
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    start_time = time()
    loss_avg = 0.0
    for x in tqdm(dataloader):
        x = x.unsqueeze(1).to(device)
        x = H(x, random.randint(0, 7))
        q = torch.randint(low=1, high=N+1, size=(gpu_num,batch_size//gpu_num), device=device)
        q_G = (torch.rand(gpu_num, batch_size//gpu_num, device=device) * q).round()
        q_DCT = q - q_G
        x_out = model(x, q_G, q_DCT)
        loss = ((x_out - x).pow(2) + 1e-6).pow(0.5).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
    scheduler.step()
    loss_avg /= iter_num
    log_data = "[%d/%d] Average loss: %f, time cost: %.2fs, cur lr is %f." % (epoch_i, end_epoch, loss_avg, time() - start_time, scheduler.get_last_lr()[0])
    print(log_data)
    with open(log_path, "a") as log_file:
        log_file.write(log_data + "\n")
    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))
    if epoch_i == 1 or epoch_i % 10 == 0:
        for cs_ratio in cs_ratio_list:
            cur_psnr, cur_ssim = test(cs_ratio)
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_epoch = epoch_i
                torch.save(model.state_dict(), "./%s/best_params.pkl" % (model_dir))
            log_data = "CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f, best PSNR is %.2f, best epoch is %d." % (cs_ratio, cur_psnr, cur_ssim, best_psnr, best_epoch)
            print(log_data)
            with open(log_path, "a") as log_file:
                log_file.write(log_data + "\n")
        torch.save(model.state_dict(), "./%s/final_params.pkl" % (model_dir))