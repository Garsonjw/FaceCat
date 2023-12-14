


def train(config):



  criterion = {'softmax': nn.CrossEntropyLoss().cuda()}

  for iter_num in range(iter_num_start, 4000 + 1):

    ######### forward #########
    classifier_label_out , feature = net1(input_data, True)
    cls_loss = criterion['softmax'](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)
    total_loss = cls_loss

    total_loss.backward()
    optimizer1.step()
    optimizer1.zero_grad()

    loss_classifier.update(cls_loss.item())

    acc = accuracy(
        classifier_label_out.narrow(0, 0, input_data.size(0)),
        source_label,
        topk=(1,))
    classifer_top1.update(acc[0])


    if (iter_num != 0 and (iter_num + 1) % (iter_per_epoch) == 0):
        valid_args = eval(test_dataloader, net1, True)
        # judge model according to HTER
        is_best = valid_args[3] <= best_model_HTER
        best_model_HTER = min(valid_args[3], best_model_HTER)
        threshold = valid_args[5]
        if (valid_args[3] <= best_model_HTER):
            best_model_ACC = valid_args[6]
            best_model_AUC = valid_args[4]
            best_TPR_FPR = valid_args[-1]

        save_list = [
        epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER,
        threshold
        ]

        save_checkpoint(save_list, is_best, net1,
                        os.path.join(config.op_dir, config.tgt_data + f'{train_method}_checkpoint_run_{str(config.run)}.pth.tar'))

        print('\r', end='', flush=True)
        log.write(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
            % ((iter_num + 1) / iter_per_epoch, 
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100, 
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100), time_to_str(timer() - start, 'min'), 0))
        log.write('\n')

        time.sleep(0.01)

  return best_model_HTER*100.0, best_model_AUC*100.0, best_TPR_FPR*100.0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str)
  parser.add_argument('--ckpt', type=str, default=None)
  parser.add_argument('--op_dir', type=str, default=None)
  parser.add_argument('--report_logger_path', type=str, default=None)
  parser.add_argument('--method', type=str, default=None, help='flip_it or flip_v')
  args = parser.parse_args()

  train_method = args.method

