import time

import torch
import torch.linalg as LA


from .autopgd import APGDAttack, APGDAttack_targeted
from .fab import FABAttack_PT
from .square import SquareAttack


class AutoAttack:
    def __init__(self, model, **kwargs):
        params = {
            "norm": "Linf",
            "eps": 0.031,
            "seed": None,
            "verbose": True,
            "attacks_to_run": [],
            "version": "standard"
        }
        params.update(kwargs)

        self.model = model
        self.norm = params["norm"]
        self.epsilon = params["eps"]
        self.seed = params["seed"]
        self.verbose = params["verbose"]
        self.attacks_to_run = params["attacks_to_run"]
        self.version = params["version"]

        assert self.norm in ["Linf", "L2", "L1"]
        assert self.version in ['standard', 'plus', 'rand']
        
        self.init_attacker()

    def init_attacker(self):
        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed)
        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed)
        self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            norm=self.norm, verbose=False)
        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            n_restarts=1, seed=self.seed, verbose=False, resc_schedule=False)

        if self.version == "standard":
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif self.version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print(f"{self.version} version is used with {self.norm} norm: please check")
        
        elif self.version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20
        
        else:
            raise ValueError(f"Version {self.version} not supported")

        if self.verbose:
            print(f"Using and setting parameters for {self.version} version including {', '.join(self.attacks_to_run).upper()}")
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def clean_accuracy(self, x_ori, y_ori):
        acc = (self.model(x_ori).argmax(dim=-1) == y_ori).item()/y_ori.size(0)

        if self.verbose:
            print(f"clean accuracy: {acc:.2%}")
        
        return acc

    @torch.no_grad()
    def run_standard_evaluation(self, x_ori, y_ori):
        start = time.process_time()
        flag = (self.model(x_ori).argmax(dim=-1) == y_ori)
        end = time.process_time()

        robust_accuracy = flag.sum().item() / x_ori.size(0)
        if self.verbose:
            print(f"{'':>8s} {'Correct':>8s} {'Robust':>7s} {'Time':>6s}")
            print(f"{'CLEAN':>8s} {f'{flag.sum().item()}/{x_ori.size(0)}':>8s} {robust_accuracy:7.2%} {end-start:6.1f}")

        x_adv = x_ori.detach().clone()

        for attack in self.attacks_to_run:
            if flag.sum().item() == 0:
                break

            start = time.process_time()
            x, y = x_ori[flag], y_ori[flag]
            robust_index = torch.nonzero(flag).squeeze(dim=-1)

            # run attack
            if attack == "apgd-ce":
                # apgd on cross-entropy loss
                self.apgd.loss = "ce"
                self.apgd.seed = self.get_seed()
                adv_curr = self.apgd.perturb(x, y)
            elif attack == "apgd-dlr":
                # apgd on dlr loss
                self.apgd.loss = "dlr"
                self.apgd.seed = self.get_seed()
                adv_curr = self.apgd.perturb(x, y)
            elif attack == "apgd-t":
                # targeted apgd
                self.apgd_targeted.seed = self.get_seed()
                adv_curr = self.apgd_targeted.perturb(x, y)            
            elif attack == "fab":
                # fab
                self.fab.targeted = False
                self.fab.seed = self.get_seed()
                adv_curr = self.fab.perturb(x, y)
            elif attack == "fab-t":
                # fab targeted
                self.fab.targeted = True
                self.fab.n_restarts = 1
                self.fab.seed = self.get_seed()
                adv_curr = self.fab.perturb(x, y)         
            elif attack == "square":
                # square
                self.square.seed = self.get_seed()
                adv_curr = self.square.perturb(x, y)    
            else:
                raise ValueError("Attack not supported")

            non_robust_flag = self.model(adv_curr).argmax(dim=-1) != y
            x_adv[robust_index[non_robust_flag]] = adv_curr[non_robust_flag].detach()
            flag[robust_index[non_robust_flag]] = False
            robust_accuracy = flag.sum().item() / y.size(0)

            end = time.process_time()

            if self.verbose:
                print(f"{attack.upper():>8s} {f'{flag.sum().item()}/{non_robust_flag.size(0)}':>8s} {robust_accuracy:7.2%} {end-start:6.1f}")

        # final check
        if self.verbose:
            if self.norm == "Linf":
                res = LA.norm((x_adv - x_ori).flatten(start_dim=1), dim=-1, ord=float("inf")).max(dim=-1).values
            elif self.norm == "L2":
                res = LA.norm((x_adv - x_ori).flatten(start_dim=1), dim=-1, ord=2).max(dim=-1).values
            elif self.norm == "L1":
                res = LA.norm((x_adv - x_ori).flatten(start_dim=1), dim=-1, ord=1).max(dim=-1).values
            print(f"AutoAttack robust accuracy {robust_accuracy:.2%}. Max {self.norm} perturbation: {res.max().item():.5f}")

        return x_adv

    @torch.no_grad()
    def run_standard_evaluation_individual(self, x_ori, y_ori):
        start = time.process_time()
        flag = (self.model(x_ori).argmax(dim=-1) == y_ori)
        end = time.process_time()

        robust_accuracy = flag.sum().item() / x_ori.size(0)
        if self.verbose:
            print(f"{'':>8s} {'Correct':>8s} {'Robust':>7s} {'Time':>6s}")
            print(f"{'CLEAN':>8s} {f'{flag.sum().item()}/{x_ori.size(0)}':>8s} {robust_accuracy:7.2%} {end-start:6.1f}")
                
        x_adv = x_ori.detach().clone()
        lowest = robust_accuracy

        for attack in self.attacks_to_run:
            start = time.process_time()
            x, y = x_ori, y_ori

            # run attack
            if attack == "apgd-ce":
                # apgd on cross-entropy loss
                self.apgd.loss = "ce"
                self.apgd.seed = self.get_seed()
                adv_curr = self.apgd.perturb(x, y)
            elif attack == "apgd-dlr":
                # apgd on dlr loss
                self.apgd.loss = "dlr"
                self.apgd.seed = self.get_seed()
                adv_curr = self.apgd.perturb(x, y)
            elif attack == "apgd-t":
                # targeted apgd
                self.apgd_targeted.seed = self.get_seed()
                adv_curr = self.apgd_targeted.perturb(x, y)            
            elif attack == "fab":
                # fab
                self.fab.targeted = False
                self.fab.seed = self.get_seed()
                adv_curr = self.fab.perturb(x, y)
            elif attack == "fab-t":
                # fab targeted
                self.fab.targeted = True
                self.fab.n_restarts = 1
                self.fab.seed = self.get_seed()
                adv_curr = self.fab.perturb(x, y)         
            elif attack == "square":
                # square
                self.square.seed = self.get_seed()
                adv_curr = self.square.perturb(x, y)    
            else:
                raise ValueError("Attack not supported")
    
            flag = (self.model(adv_curr).argmax(dim=-1) == y)
            robust_accuracy = torch.sum(flag).item() / x_ori.size(0)
            if robust_accuracy < lowest:
                lowest = robust_accuracy
                x_adv = adv_curr
            
            end = time.process_time()

            if self.verbose:
                print(f"{attack.upper():>8s} {f'{flag.sum().item()}/{flag.size(0)}':>8s} {robust_accuracy:7.2%} {end-start:6.1f}")
        return x_adv