elif g_pool.play.value:
            if len(player.captures):
                frame = player.captures[player.current_video].get_frame()
                img = frame.img
                if img:
                    draw_gl_texture(img)
                else:
                    player.captures[player.current_video].rewind()
                    player.current_video +=1
                    if player.current_video >= len(player.captures):
                        player.current_video = 0
                    g_pool.play.value = False
            else:
                print 'PLAYER: Warning: No Videos available to play. Please put your vidoes into a folder called "src_video" in the Capture folder.'