if self.device.type == "cuda":
            try:
                self.model = self.model.half().to(self.device)
            except Exception:
                self.model = self.model.to(self.device)
            import platform
            # --- Skip torch.compile on Windows since Triton isn't available ---
            if platform.system().lower() != "windows":
                try:
                    import triton
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("[INFO] Model compiled with torch.compile")
                except Exception as e:
                    print(f"[WARN] torch.compile skipped/failed: {e}")
            else:
                print("[INFO] Skipping torch.compile on Windows (Triton not supported)")
        else:
            self.model = self.model.to(self.device)
        self.model.eval()