class GAN(nn.Module):
    def __init(self,discriminator,generator):
        self.generator = Generator()
        self.discriminator = Discriminator()
    def forward(self,x,y_real):
        y_predict = self.generator(x)
        y = torch.cat((y_predict,y_real),0)
        return Discriminator(y)