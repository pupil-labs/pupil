We use `deploy_capture`, `deploy_player`, and `deploy_service` to create application bundles for deployment.

Since pyinstaller does not cross compile the scripts need to work on mac/linux/windows.


This process in __NOT RECOMMENDED OR REQUIRED FOR USERS__! (Adding documented user support for bundling is hard to support and user problems almost impossible to debug.)

You have a few options:
 - Use the runtime plugin-loader and a recent release bundle to customise your app. (the plugin loader will work in the bundle version just as it does when running from source.) <--RECOMMENDED
 - If you just need a recent version of the main repo bundled, raise an issue and we will bundle and release ASAP.
 - Figure out a setup that bundles by your own. (Expect a world of pain.)
