# Model sizes and trainers

In Veritate, a trainer is a small bundle that knows how to train a model at one specific size. Each trainer is a folder with a script that does the training and a manifest file that tells the dashboard its name, its size, and its default settings. When you open the Training tab, every installed trainer shows up in the list.

The core rule is: one trainer equals one size. Each trainer is standalone and is named after the size of model it produces. Because the size is fixed by the trainer you pick, there is no separate size selector in the form for these trainers.

The canonical trainer set covers a wide range of sizes. The trainers are: Veritate 10M, Veritate 80M, Veritate 200M, Veritate 400M, Veritate 800M, Veritate 1.3B, Veritate 3B, Veritate 13B, Veritate 50B, Veritate 70B, Veritate 100B, Veritate 120B, Veritate 160B, Veritate 200B, Veritate 250B, Veritate 350B, Veritate 500B, Veritate 700B, and Veritate 1T. The size in each name is the approximate parameter count, from ten million parameters up to one trillion.

The smaller trainers, such as Veritate 10M through roughly Veritate 800M, are the ones most likely to train end to end on a single consumer machine. The larger sizes exist as part of the canonical product line and target progressively more capable hardware.

Every model these trainers produce shares one architecture: a byte-level decoder with a fixed vocabulary of 256. The shape differs between sizes (number of layers, hidden dimension, feed-forward width, number of attention heads), but the model class is the same. This is why one inference engine can load any of them.

To get the latest trainers, use the Sync action in the dashboard's Settings tab. Trainers are pulled from a shared upstream source, so syncing keeps your set current.
