We use `deploy_capture` and `deploy_player` to create application bundles for deplayment.

Since pyinstaller does not cross compile the scripts need to work on mac/linux/windows.


This process in NOT RECOMMENDED OR REQUIRED FOR USERS! (Adding documented user support for bundling is hard to support and user problems almost impossible to debug.)

You have four options:

 - Send us a link to a fork and we will check it out, bundle and send it to you.
 - If you just need a recent version of the main repo bundled, raise an issue and we will bundle and release within 24h.)
 - Use the runtime plugin-loader and a recent release bundle to customise your app. (the plungin loader will work in the bundle version just as it does when running from source. )
 - Figure out a setup that bundles by your own. (Expect a world of pain.)
