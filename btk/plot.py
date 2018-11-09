import matplotlib.pyplot as plt
try:
    import scarlet
except ImportError:
    print("Warning: Scarlet not found, you will run into errors if using plot functions.")


def plot_blends(param, blend_images, blend_list, stack_centers=None, limits=None):
    """Plots blend images as RGB(g,r,i) image, sum in all bands, and RGB with centers of objects marked"""
    if stack_centers is None:
        stack_centers = [[]]*param.batch_size
    for i in range(param.batch_size):
        num = len(blend_list[i])
        images = np.transpose(blend_images[i, :, :, 1:4], axes=(2,0,1))
        norm = scarlet.display.Asinh(img=images, Q=20)
        blend_img_rgb = scarlet.display.img_to_rgb(images, norm=norm)
        plt.figure(figsize=(8,3))
        plt.subplot(131)
        plt.imshow(blend_img_rgb)
        if limits:
            plt.xlim(limits)
            plt.ylim(limits)
        plt.title("gri bands")
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(np.sum(blend_images[i, :, :, :], axis=2))
        plt.title("Sum")
        if limits:
            plt.xlim(limits)
            plt.ylim(limits)
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(blend_img_rgb)
        plt.title("%i objects with centers"%num)
        for entry in blend_list[i]:
            #plt.plot(entry['ra']/0.2 + 59.5, entry['dec']/0.2 + 59.5,'rx')
            plt.plot(entry['dx'], entry['dy'],'rx')
        if limits:
            plt.xlim(limits)
            plt.ylim(limits)
        for cent in stack_centers[i]:
            plt.plot(cent[0], cent[1],'go', fillstyle='none')
        plt.axis('off')
        plt.show()

def plot_with_isolated(param, blend_images, isolated_images,
                       blend_list, limits=None):
    """Plots blend images and isolated images of all objects in the blend as RGB(g,r,i) images"""
    for i in range(param.batch_size):
        num = param.max_number
        images = np.transpose(blend_images[i, :, :, 1:4], axes=(2,0,1))
        norm = scarlet.display.Asinh(img=images, Q=20)
        blend_img_rgb = scarlet.display.img_to_rgb(images, norm=norm)
        plt.figure(figsize=(2,2))
        plt.imshow(blend_img_rgb)
        plt.title("%i objects"%len(blend_list[i]))
        if limits:
            plt.xlim(limits)
            plt.ylim(limits)
        plt.axis('off')
        plt.show()
        plt.figure(figsize=(8,3))
        for j in range(len(blend_list[i])):
            plt.subplot(1, num, j +1 )
            iso_blend = isolated_images[i]
            images = np.transpose(iso_blend[j, :, :, 1:4], axes=(2,0,1))
            blend_img_rgb = scarlet.display.img_to_rgb(images, norm=norm)
            plt.imshow(blend_img_rgb)
            #plt.plot(blend_list[i]['ra'][j]/0.2 + 59, blend_list[i]['dec'][j]/0.2 + 59,'rx')
            if limits:
                plt.xlim(limits)
                plt.ylim(limits)
            plt.axis('off')
        plt.show()
