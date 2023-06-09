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



def apply_lora(self, lora_w, fstrategy, lora_alpha, mm_device=None, revert=False):
    foo = fstrategy
    if(not revert):
        if(getattr(self, "_lora_w", None) is not None):
            print("Reverting lora")
            if((self._lora_w is lora_w) and lora_alpha==self._lora_alpha):
                old_fstrategy = self._fstrategy
                new_fstrategy = fstrategy
                foo = lambda x:new_fstrategy(x)-old_fstrategy(x)
                
            else:
                apply_lora(
                    self,
                    self._lora_w, 
                    self._fstrategy,
                    self._lora_alpha,
                    mm_device,
                    True
                )
    print(foo, sum([foo(x/31) for x in range(32)])/32)
    with torch.no_grad():
        # lora_w = torch.load(lora_w)
        ls = list(self.w.keys())
        iterator = tqdm(ls)
        for key in iterator:
            if(key.startswith("blocks.")):
                pref = key[:-len(".weight")]
                key_A = pref+".lora_A"
                key_B = pref+".lora_B"
                lora_found = key_A in lora_w and key_B in lora_w
                int8 = key+"_mx" in self.w
                if(lora_found and not int8):
                    w = self.w[key]
                    layer_id = int(key.split('.')[1]) if ('blocks.' in key) else 0
                    layer_depth = layer_id/(self.args.n_layer-1)
                    lora_ratio = foo(layer_depth)
                    if(lora_ratio==0):
                        continue
                    if(revert):
                        lora_ratio = -lora_ratio
                    lora_A = lora_w[key_A].to(device=mm_device, dtype=w.dtype)
                    lora_B = lora_w[key_B].to(device=mm_device, dtype=w.dtype)
                    if (mm_device is None):
                        device = w.device
                    else:
                        device = mm_device
                    lora_rank = lora_A.shape[0]
                    deltaw = (lora_B@lora_A)*lora_alpha/lora_rank*lora_ratio
                    
                    if(deltaw.shape!=w.shape):
                        shape0 = list(deltaw.shape)[::-1]
                        shape1 = list(w.shape)
                        if(shape0==shape1):
                            deltaw = deltaw.T
                        else:
                            raise Exception("Shape does not match")
                        
                    if(self.RESCALE_LAYER):
                        if 'att.output.weight' in key:
                            deltaw = deltaw/(2 ** int(layer_id // self.RESCALE_LAYER))
                        if 'ffn.value.weight' in key:
                            deltaw = deltaw/(2 ** int(layer_id // self.RESCALE_LAYER))

                    if(device!=w.device):
                        deltaw = deltaw.to(device=w.device)
                    
                    self.w[key]+=deltaw
                    delta = deltaw.abs().sum()
                    iterator.desc = "%s: %.2f"%(key, lora_ratio)
                    # print("Merged %s with alpha/rank*ratio = %.2f/%.2f*%.2f = %.3f, |delta w|=%.3f"%(key, lora_alpha, lora_rank, lora_ratio, lora_alpha/lora_rank*lora_ratio, delta), end="\n")

    self._lora_w = lora_w
    self._fstrategy = fstrategy
    self._lora_alpha = lora_alpha
    print()
