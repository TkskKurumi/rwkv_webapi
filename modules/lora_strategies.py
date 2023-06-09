import torch
from tqdm import tqdm
def constant(ratio):
    def inner(depth):
        return ratio
    return inner
def poly(*args):
    xys = sorted(args)
    xs = [x for x, y in xys]
    ys = [y for x, y in xys]
    def inner(x):
        if(x<=xs[0]):
            return ys[0]
        if(x>=xs[-1]):
            return ys[-1]
        for idx, xy0 in enumerate(xys):
            x0, y0 = xy0
            x1, y1 = xys[idx+1]
            if(x0<=x and x<=x1):
                ratio = (x-x0)/(x1-x0)
                return y0*(1-ratio) + y1*ratio
        assert False
    return inner

def get_fstrategy(x):
    return eval(x)



def apply_lora(w, lora_w, fstrategy, lora_alpha, mm_device=None, model=None):
    foo = fstrategy
    print(foo, sum([foo(x/31) for x in range(32)])/32)
    if(model is not None):
        n_layer = model.args.n_layer
        RESCALE_LAYER = model.RESCALE_LAYER
    else:
        n_layer = 32
        RESCALE_LAYER = 0
    with torch.no_grad():
        # lora_w = torch.load(lora_w)
        ls = list(w.keys())
        iterator = tqdm(ls)
        for key in iterator:
            if(key.startswith("blocks.")):
                pref = key[:-len(".weight")]
                key_A = pref+".lora_A"
                key_B = pref+".lora_B"
                lora_found = key_A in lora_w and key_B in lora_w
                int8 = key+"_mx" in w
                if(lora_found and not int8):
                    layer_w = w[key]
                    layer_id = int(key.split('.')[1]) if ('blocks.' in key) else 0
                    layer_depth = layer_id/(n_layer-1)
                    lora_ratio = foo(layer_depth)
                    if(lora_ratio==0):
                        continue
                    lora_A = lora_w[key_A].to(device=mm_device, dtype=layer_w.dtype)
                    lora_B = lora_w[key_B].to(device=mm_device, dtype=layer_w.dtype)
                    if (mm_device is None):
                        device = layer_w.device
                    else:
                        device = mm_device
                    lora_rank = lora_A.shape[0]
                    deltaw = (lora_B@lora_A)*lora_alpha/lora_rank*lora_ratio
                    
                    if(deltaw.shape!=layer_w.shape):
                        shape0 = list(deltaw.shape)[::-1]
                        shape1 = list(layer_w.shape)
                        if(shape0==shape1):
                            deltaw = deltaw.T
                        else:
                            raise Exception("Shape does not match")
                        
                    if(RESCALE_LAYER):
                        if 'att.output.weight' in key:
                            deltaw = deltaw/(2 ** int(layer_id // RESCALE_LAYER))
                        if 'ffn.value.weight' in key:
                            deltaw = deltaw/(2 ** int(layer_id // RESCALE_LAYER))

                    if(device!=layer_w.device):
                        deltaw = deltaw.to(device=layer_w.device)
                    
                    w[key]+=deltaw
                    delta = deltaw.abs().sum()
                    iterator.desc = "%s: %.2f"%(key, lora_ratio)
                    # print("Merged %s with alpha/rank*ratio = %.2f/%.2f*%.2f = %.3f, |delta w|=%.3f"%(key, lora_alpha, lora_rank, lora_ratio, lora_alpha/lora_rank*lora_ratio, delta), end="\n")

    print()

class LoRA:
    def __init__(self, model, lora_pth, strategy, lora_alpha, mm_device="cuda", auto_revert=False):

        self.lora_pth = lora_pth

        self.model = model
        with torch.no_grad():
            self.w = torch.load(lora_pth, map_location="cpu")
        self.strategy = strategy
        self.lora_alpha = lora_alpha
        self.mm_device = mm_device
        self.auto_revert = auto_revert

        self.applied = False

    def apply(self):
        apply_lora(self.model.w, self.w, self.strategy, self.lora_alpha, mm_device=self.mm_device, model=self.model)
        self.applied = True
    def revert(self):
        delta_strategy = lambda x:-self.strategy(x)
        apply_lora(self.model.w, self.w, self.strategy, self.lora_alpha, mm_device=self.mm_device, model=self.model)
        self.applied = False
    def change_strategy(self, strategy):
        if(self.applied):
            delta_strategy = lambda x:strategy(x) - self.strategy(x)
            apply_lora(self.model.w, self.w, delta_strategy, self.lora_alpha, mm_device=self.mm_device, model=self.model)
            self.strategy = strategy
        else:
            apply_lora(self.model.w, self.w, strategy, self.lora_alpha, mm_device=self.mm_device, model=self.model)
    
    def __del__(self):
        if(self.applied):
            print("WARNING: LoRA", self.lora_pth, "is not reverted before exit")
            if(self.auto_revert):
                self.revert()